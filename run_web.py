"""Start the AI Customer & Business Intelligence web app.

Usage:
    python run_web.py

Environment variables:
    HOST=0.0.0.0        Bind address (default 127.0.0.1)
    PORT=5000           Port
    FLASK_ENV=production
    SECRET_KEY=...      Session signing key (auto-generated if omitted)
    DATABASE_URL=...    e.g. mysql+pymysql://user:pass@host/db
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "backend" / "routes" / "routes"

for entry in (str(APP_DIR), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from app import create_app  # noqa: E402

app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_ENV", "development").lower() != "production"

    print("\nAI Customer & Business Intelligence web app")
    print(f"  URL:      http://{host}:{port}")
    print(f"  Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print("  Stop with Ctrl+C\n")

    app.run(host=host, port=port, debug=debug)
