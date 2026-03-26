"""OpenEnv-compatible ASGI entrypoint."""

from main import app


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
    # main()
