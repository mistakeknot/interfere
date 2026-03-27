"""Entry point: ``python -m server`` or ``uv run python -m server``."""

from __future__ import annotations

import argparse
import sys

import uvicorn

from .main import create_app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="interfere",
        description="Local MLX-LM inference server for Apple Silicon",
    )
    parser.add_argument(
        "--port", type=int, default=8421, help="Port to listen on (default: 8421)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Start in dry-run mode (fake tokens, no MLX)",
    )
    parser.add_argument(
        "--preload",
        type=str,
        default=None,
        help="Model to preload at startup (e.g., mlx-community/Qwen3.5-35B-A3B-4bit)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    app = create_app(dry_run=args.dry_run)

    if args.preload and not args.dry_run:
        import httpx

        # Start uvicorn in a thread, preload model via HTTP, then serve
        import threading

        server = uvicorn.Server(
            uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server to be ready
        import time

        for _ in range(30):
            try:
                r = httpx.get(f"http://{args.host}:{args.port}/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)

        # Preload model
        print(f"Preloading model: {args.preload}")
        try:
            r = httpx.post(
                f"http://{args.host}:{args.port}/v1/models/load",
                json={"model": args.preload},
                timeout=180.0,
            )
            if r.status_code == 200:
                print(f"Model preloaded: {args.preload}")
            else:
                print(f"Preload failed: {r.text}")
        except Exception as e:
            print(f"Preload error: {e}")

        thread.join()
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
