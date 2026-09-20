from __future__ import annotations

import argparse
import os

import uvicorn

from antenna_service.api import app


def run() -> None:
    parser = argparse.ArgumentParser(description="Antenna range local control service")
    parser.add_argument("--host", default=os.environ.get("ANTENNA_SERVICE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ANTENNA_SERVICE_PORT", "18765")))
    parser.add_argument("--log-level", default=os.environ.get("ANTENNA_SERVICE_LOG_LEVEL", "info"))
    arguments = parser.parse_args()
    # Pass the application object so frozen Windows builds include the complete API
    # module graph; a string-only import target can be invisible to PyInstaller.
    uvicorn.run(
        app,
        host=arguments.host,
        port=arguments.port,
        log_level=arguments.log_level,
        reload=False,
    )


if __name__ == "__main__":
    run()
