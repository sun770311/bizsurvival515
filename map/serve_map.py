"""
Mandatory Steps:
  export MAPBOX_PUBLIC_TOKEN="pk.XXXX..."
  python serve_map.py --port 8000
View map locally at:
  http://localhost:8000
"""

from __future__ import annotations

import os
from pathlib import Path

import argparse
from flask import Flask, Response, abort, send_from_directory


def create_app(map_dir: Path, data_dir: Path, token: str) -> Flask:
    app = Flask(__name__)

    @app.route("/config.js")
    def config_js() -> tuple[str, int, dict]:
        body = f'window.__CONFIG__ = {{ MAPBOX_PUBLIC_TOKEN: "{token}" }};\n'
        return body, 200, {"Content-Type": "application/javascript; charset=utf-8"}

    @app.route("/data/<path:filename>")
    def serve_data(filename: str) -> Response:
        target = (data_dir / filename).resolve()
        try:
            target.relative_to(data_dir)
        except ValueError:
            abort(403)
        if not target.exists() or not target.is_file():
            abort(404)
        return send_from_directory(data_dir, filename)

    @app.route("/")
    def index() -> Response:
        return send_from_directory(map_dir, "index.html")

    @app.route("/<path:filename>")
    def serve_static(filename: str) -> Response:
        target = (map_dir / filename).resolve()
        try:
            target.relative_to(map_dir)
        except ValueError:
            abort(403)
        if not target.exists() or not target.is_file():
            abort(404)
        return send_from_directory(map_dir, filename)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    token = os.getenv("MAPBOX_PUBLIC_TOKEN")
    if not token:
        raise SystemExit(
            "Missing MAPBOX_PUBLIC_TOKEN env var.\n"
            "Set it like:\n"
            '   export MAPBOX_PUBLIC_TOKEN="pk.XXXX..."\n'
        )

    map_dir = Path(__file__).resolve().parent
    data_dir = (map_dir / "../data").resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    app = create_app(map_dir, data_dir, token)

    print(f"Serving on http://localhost:{args.port}")
    print(f" - site root: {map_dir}")
    print(f" - data root: {data_dir}  (mounted at /data/)")
    print(" - config:    /config.js")

    app.run(host="", port=args.port, debug=False)


if __name__ == "__main__":
    main()
