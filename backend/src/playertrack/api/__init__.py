"""HTTP API for running analyses from the web frontend."""

from .app import create_app

__all__ = ["create_app"]
