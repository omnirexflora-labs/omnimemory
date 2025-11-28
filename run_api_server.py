#!/usr/bin/env python3
"""
OmniMemory API Server Runner

Simple script to run the FastAPI server with uvicorn.
"""

import uvicorn
from omnimemory.api.server import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False,
    )
