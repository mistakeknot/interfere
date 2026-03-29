"""Flash-MoE subprocess proxy for 700B+ MoE models.

Flash-MoE is a compiled C binary (metal_infer/infer) that runs its own Metal
GPU context and exposes an OpenAI-compatible HTTP/SSE API via --serve PORT.

This module manages the binary as a subprocess and proxies requests through
HTTP, presenting the same interface as MetalWorker so the HTTP layer in
main.py can use either backend interchangeably.

Unlike MetalWorker (which uses multiprocessing.Queue IPC), FlashMoeWorker
talks HTTP to the child process. This is appropriate because flash-moe
handles its own tokenization, KV cache, and generation loop.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass

import httpx

log = logging.getLogger("interfere.flashmoe_worker")

# Default path — overridden via CLI
_DEFAULT_BINARY = os.path.expanduser("~/projects/flash-moe/metal_infer/infer")

# Watchdog defaults (mirroring MetalWorker's pattern)
_WATCHDOG_POLL_INTERVAL = 3.0
_MAX_CONSECUTIVE_RESTARTS = 3
_INITIAL_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 30.0

# How long to wait for the binary to start serving.
# flash-moe's serve_loop pre-caches the system prompt through all 60 layers
# before accepting HTTP connections. For a 397B model streaming experts from
# SSD, this takes 60-120s. Set generous timeout accordingly.
_STARTUP_TIMEOUT_S = 300.0
_STARTUP_POLL_INTERVAL = 2.0

# HTTP client timeout for generation (flash-moe can take a while on first token)
_CONNECT_TIMEOUT = 5.0
_PREFILL_TIMEOUT = 120.0  # first-token timeout (includes prefill of long prompts)


def _find_free_port() -> int:
    """Find an available TCP port for the flash-moe subprocess."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class FlashMoeCrashInfo:
    """Record of a flash-moe subprocess crash."""

    timestamp: float
    exit_code: int | None
    restart_attempt: int


class FlashMoeWorker:
    """Manages a flash-moe binary subprocess and proxies inference via HTTP.

    Presents a similar interface to MetalWorker:
    - health() → dict
    - generate() → Generator[str]
    - shutdown()
    - is_alive, is_degraded, is_restarting properties

    Does NOT support probe() — flash-moe has no logprobs/confidence output.
    Cascade logic should treat flash-moe as a terminal (always-accept) tier.
    """

    def __init__(
        self,
        binary_path: str = _DEFAULT_BINARY,
        model_path: str = "",
        port: int = 0,
        extra_args: list[str] | None = None,
        enable_watchdog: bool = True,
        malloc_cache: int = 0,
        predict: bool = False,
    ) -> None:
        self._binary_path = binary_path
        self._model_path = model_path
        self._port = port or _find_free_port()
        self._extra_args = extra_args or []
        self._enable_watchdog = enable_watchdog
        self._malloc_cache = malloc_cache
        self._predict = predict

        self._process: subprocess.Popen | None = None
        self._generate_lock = threading.Lock()

        # Crash recovery state
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self._restart_count: int = 0
        self._consecutive_crashes: int = 0
        self._last_crash: FlashMoeCrashInfo | None = None
        self._crash_history: list[FlashMoeCrashInfo] = []
        self._restarting = threading.Event()
        self._degraded = False

        # Last generation metrics (populated after each generate())
        self._last_generation_metrics: dict = {}

        # Persistent HTTP client
        self._client: httpx.Client | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the flash-moe binary subprocess."""
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError("already running")

        if not os.path.isfile(self._binary_path):
            raise FileNotFoundError(f"flash-moe binary not found: {self._binary_path}")

        if not self._model_path:
            raise ValueError("model_path is required")

        cmd = [
            self._binary_path,
            "--model",
            self._model_path,
            "--serve",
            str(self._port),
        ]
        if self._malloc_cache > 0:
            cmd += ["--malloc-cache", str(self._malloc_cache)]
        if self._predict:
            cmd.append("--predict")
        cmd += self._extra_args

        log.info("starting flash-moe: %s", " ".join(cmd))

        # Run from the binary's directory so it finds shaders.metal
        # (flash-moe looks for shaders.metal in CWD or metal_infer/).
        binary_dir = os.path.dirname(self._binary_path)

        # Merge stdout+stderr so we see all flash-moe output in one stream.
        # Flash-moe writes progress info to stderr during loading.
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=binary_dir or None,
        )

        # Background thread to drain and log subprocess output
        self._log_thread = threading.Thread(
            target=self._drain_output,
            daemon=True,
            name="interfere-flashmoe-log",
        )
        self._log_thread.start()

        # Create HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT,
                read=_PREFILL_TIMEOUT,
                write=5.0,
                pool=5.0,
            ),
        )

        # Wait for the serve endpoint to come alive
        self._wait_for_ready()

        log.info(
            "flash-moe ready: pid=%d port=%d model=%s",
            self._process.pid,
            self._port,
            self._model_path,
        )

        if self._enable_watchdog and self._watchdog_thread is None:
            self._start_watchdog()

    def _drain_output(self) -> None:
        """Read flash-moe subprocess output and forward to our logger."""
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    log.info("[flash-moe] %s", line)
        except (ValueError, OSError):
            pass  # pipe closed

    def _wait_for_ready(self) -> None:
        """Block until flash-moe's /health endpoint responds or timeout.

        flash-moe's serve_loop binds the socket then pre-caches the system
        prompt (60 layers, 60-120s for 397B) before calling accept(). During
        this time TCP connect may succeed (socket is listening) but the HTTP
        request will hang. We use a short connect+read timeout so we don't
        block on a single attempt.
        """
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        attempt = 0
        while time.monotonic() < deadline:
            # Check if process died during startup
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"flash-moe exited during startup (code {self._process.returncode})"
                )
            try:
                resp = httpx.get(
                    f"{self.base_url}/health",
                    timeout=httpx.Timeout(connect=2.0, read=5.0, write=2.0, pool=2.0),
                )
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                # Port not yet bound — process still loading
                pass
            except (httpx.ReadTimeout, httpx.TimeoutException):
                # TCP connected but no response yet — system prompt caching
                if attempt % 10 == 0:
                    elapsed = int(time.monotonic() - (deadline - _STARTUP_TIMEOUT_S))
                    log.info(
                        "flash-moe loading (system prompt cache in progress, %ds elapsed)...",
                        elapsed,
                    )
            attempt += 1
            time.sleep(_STARTUP_POLL_INTERVAL)

        raise TimeoutError(
            f"flash-moe did not become ready within {_STARTUP_TIMEOUT_S}s"
        )

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop the flash-moe subprocess."""
        self._stop_watchdog()

        if self._client is not None:
            self._client.close()
            self._client = None

        if self._process is None:
            return

        if self._process.poll() is None:
            # Send SIGTERM for graceful shutdown
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)

        self._process = None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def is_restarting(self) -> bool:
        return self._restarting.is_set()

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    @property
    def restart_count(self) -> int:
        return self._restart_count

    @property
    def last_crash(self) -> FlashMoeCrashInfo | None:
        return self._last_crash

    @property
    def crash_history(self) -> list[FlashMoeCrashInfo]:
        return list(self._crash_history)

    @property
    def last_generation_metrics(self) -> dict:
        return self._last_generation_metrics

    def reset_consecutive_crashes(self) -> None:
        if self._consecutive_crashes > 0:
            self._consecutive_crashes = 0

    # -- commands ------------------------------------------------------------

    def health(self, timeout: float = 5.0) -> dict:
        """Ping flash-moe and return health info.

        Returns a dict compatible with MetalWorker's health response shape:
        {"status": "ready", "pid": ..., "loaded_models": [...]}
        """
        if not self.is_alive():
            return {"status": "down", "pid": None, "loaded_models": []}

        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=timeout)
            data = resp.json()
            return {
                "status": "ready",
                "pid": self._process.pid if self._process else None,
                "model": data.get("model", "unknown"),
                "loaded_models": [data.get("model", "flash-moe")],
                "backend": "flash-moe",
                "port": self._port,
            }
        except Exception as exc:
            return {
                "status": "unresponsive",
                "pid": self._process.pid if self._process else None,
                "loaded_models": [],
                "error": str(exc),
            }

    def generate(
        self,
        model_name: str = "",
        messages: list[dict] | None = None,
        prompt: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        timeout: float = 300.0,
        **kwargs,  # absorb kv_bits etc. that flash-moe doesn't support
    ) -> Generator[str, None, None]:
        """Stream generated tokens from flash-moe via HTTP/SSE.

        Interface matches MetalWorker.generate() — yields decoded text strings.
        Concurrent calls are serialized by _generate_lock.
        """
        with self._generate_lock:
            if not self.is_alive():
                raise RuntimeError("flash-moe subprocess not running")

            # Build the request body.
            # flash-moe expects messages array; if only prompt given, wrap it.
            if messages:
                body_messages = messages
            elif prompt:
                body_messages = [{"role": "user", "content": prompt}]
            else:
                raise ValueError("messages or prompt required")

            body = {
                "messages": body_messages,
                "max_tokens": max_tokens,
                "stream": True,
            }

            t0 = time.monotonic()
            completion_tokens = 0

            try:
                # Use httpx streaming to consume SSE events
                with httpx.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    timeout=httpx.Timeout(
                        connect=_CONNECT_TIMEOUT,
                        read=timeout,  # per-chunk read timeout
                        write=5.0,
                        pool=5.0,
                    ),
                ) as response:
                    if response.status_code != 200:
                        error_body = response.read().decode("utf-8", errors="replace")
                        raise RuntimeError(
                            f"flash-moe returned {response.status_code}: {error_body}"
                        )

                    for line in response.iter_lines():
                        if not line:
                            continue

                        # SSE format: "data: {...}" or "data: [DONE]"
                        if not line.startswith("data: "):
                            continue

                        payload = line[6:]  # strip "data: " prefix

                        if payload == "[DONE]":
                            break

                        try:
                            chunk = json.loads(payload)
                        except (ValueError, json.JSONDecodeError):
                            continue

                        # Extract token content from OpenAI delta format
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        finish = choices[0].get("finish_reason")

                        if content:
                            completion_tokens += 1
                            yield content

                        if finish == "stop":
                            break

            except httpx.ReadTimeout:
                raise TimeoutError(f"flash-moe generation timed out after {timeout}s")

            elapsed = time.monotonic() - t0
            tps = completion_tokens / elapsed if elapsed > 0 else 0

            self._last_generation_metrics = {
                "generation_tps": round(tps, 1),
                "prompt_tps": 0,  # flash-moe doesn't expose prefill metrics via SSE
                "peak_memory_gb": 0,  # not available via HTTP
                "early_exit_rate": 0,
                "mean_confidence": 0,
                "tokens_generated": completion_tokens,
                "backend": "flash-moe",
                "elapsed_s": round(elapsed, 3),
            }

            log.info(
                "flash-moe generate: tokens=%d time=%.2fs tps=%.1f",
                completion_tokens,
                elapsed,
                tps,
            )

    # -- watchdog ------------------------------------------------------------

    def restart(self) -> None:
        """Kill and restart the flash-moe subprocess."""
        self._restarting.set()
        try:
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=5.0)
            self._process = None

            if self._client is not None:
                self._client.close()
                self._client = None

            # Re-pick port in case the old one is stuck in TIME_WAIT
            self._port = _find_free_port()
            self.start()
            self._restart_count += 1
            log.info(
                "flash-moe restarted (attempt %d, total restarts: %d)",
                self._consecutive_crashes,
                self._restart_count,
            )
        finally:
            self._restarting.clear()

    def _start_watchdog(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="interfere-flashmoe-watchdog",
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5.0)
            self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        """Background thread: detect flash-moe crashes and auto-restart."""
        while not self._watchdog_stop.is_set():
            self._watchdog_stop.wait(timeout=_WATCHDOG_POLL_INTERVAL)
            if self._watchdog_stop.is_set():
                break

            if self._degraded or self._restarting.is_set():
                continue

            if self._process is not None and self._process.poll() is not None:
                exit_code = self._process.returncode
                crash = FlashMoeCrashInfo(
                    timestamp=time.time(),
                    exit_code=exit_code,
                    restart_attempt=self._consecutive_crashes + 1,
                )
                self._last_crash = crash
                self._crash_history.append(crash)
                self._consecutive_crashes += 1

                log.warning(
                    "flash-moe crashed: exit_code=%s (consecutive: %d/%d)",
                    exit_code,
                    self._consecutive_crashes,
                    _MAX_CONSECUTIVE_RESTARTS,
                )

                if self._consecutive_crashes > _MAX_CONSECUTIVE_RESTARTS:
                    log.error("max restarts exceeded — entering degraded mode")
                    self._degraded = True
                    continue

                backoff = min(
                    _INITIAL_BACKOFF_S * (2 ** (self._consecutive_crashes - 1)),
                    _MAX_BACKOFF_S,
                )
                log.info("waiting %.1fs before restart", backoff)
                self._watchdog_stop.wait(timeout=backoff)
                if self._watchdog_stop.is_set():
                    break

                try:
                    self.restart()
                except Exception as exc:
                    log.error("flash-moe restart failed: %s", exc)
