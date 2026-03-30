"""
Root-level inference entrypoint for platform compatibility checks.

Some evaluators require an `inference.py` file at repository root. This file
re-exports the ASGI app and provides a runnable `main()` entrypoint.
"""

from server.app import app, main


if __name__ == "__main__":
    main()
