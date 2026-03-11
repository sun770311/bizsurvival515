"""Module to serve the interactive Mapbox visualization using Flask."""

from pathlib import Path
from flask import Flask, send_from_directory, abort

APP_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(APP_DIR), static_url_path="")

REQUIRED_FILES = ["index.html", "app.js", "businesses.geojson"]


def ensure_file_exists(filename: str) -> None:
    """Check if a required static file exists, abort with 404 if not."""
    file_path = APP_DIR / filename
    if not file_path.exists():
        abort(404, description=f"Required file not found: {filename}")


@app.route("/")
def index():
    """Serve the main index.html file."""
    ensure_file_exists("index.html")
    return send_from_directory(APP_DIR, "index.html")


@app.route("/app.js")
def app_js():
    """Serve the app.js JavaScript application logic."""
    ensure_file_exists("app.js")
    return send_from_directory(APP_DIR, "app.js", mimetype="application/javascript")


@app.route("/businesses.geojson")
def geojson():
    """Serve the generated GeoJSON dataset for the map."""
    ensure_file_exists("businesses.geojson")
    return send_from_directory(APP_DIR, "businesses.geojson", mimetype="application/geo+json")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve any other static files in the directory."""
    file_path = APP_DIR / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(APP_DIR, filename)
    abort(404, description=f"File not found: {filename}")


if __name__ == "__main__":
    MISSING = [f for f in REQUIRED_FILES if not (APP_DIR / f).exists()]
    if MISSING:
        print("Missing required files:")
        for missing_file in MISSING:
            print(f" - {missing_file}")
    else:
        print("Serving files from:", APP_DIR)
        print("Open: http://127.0.0.1:8000")

    app.run(host="127.0.0.1", port=8000, debug=True)
