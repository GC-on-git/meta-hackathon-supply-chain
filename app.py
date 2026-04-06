# Top-level ASGI entry for Docker / local `uvicorn app:app`.

from server.app import app, main

__all__ = ["app", "main"]
