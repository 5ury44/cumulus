#!/usr/bin/env python3
"""
Start the Cumulus worker server
"""

import argparse
import uvicorn
from worker.server import create_app


def main():
    parser = argparse.ArgumentParser(description="Start Cumulus worker server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    app = create_app()
    
    print(f"🚀 Starting Cumulus worker server on {args.host}:{args.port}")
    print(f"Workers: {args.workers}")
    print(f"Reload: {args.reload}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
