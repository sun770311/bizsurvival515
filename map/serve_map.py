from pathlib import Path
from flask import Flask, send_from_directory, abort

APP_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(APP_DIR), static_url_path="")

REQUIRED_FILES = ["index.html", "app.js", "businesses.geojson"]


def ensure_file_exists(filename: str) -> None:
    file_path = APP_DIR / filename
    if not file_path.exists():
        abort(404, description=f"Required file not found: {filename}")


@app.route("/")
def index():
    ensure_file_exists("index.html")
    return send_from_directory(APP_DIR, "index.html")


@app.route("/app.js")
def app_js():
    ensure_file_exists("app.js")
    return send_from_directory(APP_DIR, "app.js", mimetype="application/javascript")


@app.route("/businesses.geojson")
def geojson():
    ensure_file_exists("businesses.geojson")
    return send_from_directory(APP_DIR, "businesses.geojson", mimetype="application/geo+json")


@app.route("/<path:filename>")
def static_files(filename):
    file_path = APP_DIR / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory(APP_DIR, filename)
    abort(404, description=f"File not found: {filename}")


if __name__ == "__main__":
    missing = [f for f in REQUIRED_FILES if not (APP_DIR / f).exists()]
    if missing:
        print("Missing required files:")
        for f in missing:
            print(f" - {f}")
    else:
        print("Serving files from:", APP_DIR)
        print("Open: http://127.0.0.1:8000")

    app.run(host="127.0.0.1", port=8000, debug=True)