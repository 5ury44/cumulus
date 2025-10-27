#!/usr/bin/env python3
"""
Simple Cumulus CLI for serving and Chronos helpers.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional

from cumulus.worker.server import create_app


def _run_uvicorn(host: str, port: int, workers: int, reload: bool) -> None:
    try:
        import uvicorn
    except Exception as e:
        print(f"uvicorn is required to serve: {e}", file=sys.stderr)
        sys.exit(1)

    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
    )


def cmd_serve(args: argparse.Namespace) -> None:
    _run_uvicorn(args.host, args.port, args.workers, args.reload)


def _resolve_chronos_path() -> Optional[str]:
    # Align with CumulusManager logic
    env_path = os.getenv("CUMULUS_CHRONOS_PATH")
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    # Vendored path
    here = os.path.dirname(__file__)
    vendored = os.path.join(here, "chronos_vendor", "bin", "chronos_cli")
    if os.path.isfile(vendored) and os.access(vendored, os.X_OK):
        return vendored

    # System path fallback
    return "/usr/local/bin/chronos_cli"


def cmd_chronos_path(args: argparse.Namespace) -> None:
    print(_resolve_chronos_path())


def cmd_chronos(args: argparse.Namespace) -> None:
    cli = _resolve_chronos_path()
    if not cli or not os.path.isfile(cli):
        print("chronos_cli not found. Build vendored Chronos or set CUMULUS_CHRONOS_PATH.", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/lib:' + env.get('LD_LIBRARY_PATH', '')

    cmd = [cli] + args.rest
    try:
        rc = subprocess.call(cmd, env=env)
        sys.exit(rc)
    except FileNotFoundError:
        print(f"Failed to execute {cli}", file=sys.stderr)
        sys.exit(1)


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(prog="cumulus-cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="Start the Cumulus worker HTTP server")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--workers", type=int, default=1)
    sp.add_argument("--reload", action="store_true")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("chronos-path", help="Print resolved chronos_cli path")
    sp.set_defaults(func=cmd_chronos_path)

    sp = sub.add_parser("chronos", help="Proxy commands to chronos_cli (vendored or system)")
    sp.add_argument("rest", nargs=argparse.REMAINDER)
    sp.set_defaults(func=cmd_chronos)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


