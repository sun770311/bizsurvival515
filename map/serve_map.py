"""
Mandatory Steps:
  export MAPBOX_PUBLIC_TOKEN="pk.XXXX..."
  python serve_map.py --port 8000
View map locally at:
  http://localhost:8000
"""

from __future__ import annotations

import argparse
import http.server
import mimetypes
import os
import socketserver
from pathlib import Path
from urllib.parse import unquote, urlparse


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

    map_dir = Path(__file__).resolve().parent   # .../map
    data_dir = (map_dir / "../data").resolve()  # .../data

    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    class DualRootHandler(http.server.BaseHTTPRequestHandler):
        server_version = "DualRootHTTP/0.2"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)

            if path == "/config.js":
                body = f'window.__CONFIG__ = {{ MAPBOX_PUBLIC_TOKEN: "{token}" }};\n'.encode("utf-8")
                return self._send_bytes(200, body, "application/javascript; charset=utf-8")

            if path.startswith("/data/"):
                rel = path[len("/data/") :]
                target = (data_dir / rel).resolve()
                if not self._is_within(target, data_dir):
                    return self._send_text(403, "Forbidden")
                return self._serve_file(target)

            if path == "/" or path == "":
                return self._serve_file(map_dir / "index.html")

            rel = path.lstrip("/")
            target = (map_dir / rel).resolve()
            if not self._is_within(target, map_dir):
                return self._send_text(403, "Forbidden")
            return self._serve_file(target)

        def _is_within(self, target: Path, root: Path) -> bool:
            try:
                target.relative_to(root)
                return True
            except ValueError:
                return False

        def _serve_file(self, fp: Path) -> None:
            if not fp.exists() or fp.is_dir():
                return self._send_text(404, "Not found")

            ctype, _ = mimetypes.guess_type(str(fp))
            ctype = ctype or "application/octet-stream"

            data = fp.read_bytes()
            return self._send_bytes(200, data, ctype)

        def _send_text(self, code: int, text: str) -> None:
            body = (text + "\n").encode("utf-8")
            return self._send_bytes(code, body, "text/plain; charset=utf-8")

        def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

    with socketserver.TCPServer(("", args.port), DualRootHandler) as httpd:
        print(f"Serving on http://localhost:{args.port}")
        print(f" - site root: {map_dir}")
        print(f" - data root: {data_dir}  (mounted at /data/)")
        print(" - config:    /config.js")
        httpd.serve_forever()


if __name__ == "__main__":
    main()