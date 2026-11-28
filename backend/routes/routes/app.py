"""Flask application factory for the AI Business Intelligence web app.

Run from the project root with:
    python run_web.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parents[3]

for entry in (str(HERE), str(BASE_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from flask import Flask, g, jsonify, render_template, request  # noqa: E402
from werkzeug.middleware.proxy_fix import ProxyFix  # noqa: E402

import admin  # noqa: E402
import api  # noqa: E402
import auth  # noqa: E402
import views  # noqa: E402
from config import Config, get_config, INSTANCE_DIR  # noqa: E402
from database import db  # noqa: E402
from security import (  # noqa: E402
    get_csrf_token,
    load_current_user,
    validate_csrf,
)

NAV_ITEMS = [
    {"endpoint": "views.overview", "label": "Overview", "icon": "🏠"},
    {"endpoint": "views.sales", "label": "Sales", "icon": "💰"},
    {"endpoint": "views.customers", "label": "Customers", "icon": "👥"},
    {"endpoint": "views.demand", "label": "Product Demand", "icon": "📦"},
    {"endpoint": "views.sentiment", "label": "Sentiment", "icon": "💬"},
    {"endpoint": "views.live", "label": "Live Prediction", "icon": "🔮"},
]

ADMIN_NAV_ITEM = {
    "endpoint": "admin.users",
    "label": "Users",
    "icon": "🛡️",
}


def create_app(config_object=None):
    app = Flask(
        __name__,
        template_folder=str(Config.TEMPLATE_DIR),
        static_folder=str(Config.STATIC_DIR),
        static_url_path="/static",
        instance_path=str(INSTANCE_DIR),
    )

    app.config.from_object(config_object or get_config())

    # Render (and most PaaS hosts) terminate TLS at a proxy in front of the
    # app. Trust one hop of X-Forwarded-* headers so redirects and cookies
    # see the original HTTPS scheme and host.
    if os.environ.get("RENDER"):
        app.wsgi_app = ProxyFix(
            app.wsgi_app, x_for=1, x_proto=1, x_host=1
        )

    db.init_app(app)

    with app.app_context():
        db.create_all()

    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.register_blueprint(admin.bp)
    app.register_blueprint(api.bp)

    @app.before_request
    def _before_request():
        load_current_user()
        return validate_csrf()

    @app.context_processor
    def _inject_globals():
        user = g.get("user")
        return {
            "current_user": user,
            "csrf_token": get_csrf_token,
            "app_name": app.config["APP_NAME"],
            "nav_items": NAV_ITEMS,
            "admin_nav_item": ADMIN_NAV_ITEM,
            "current_year": datetime.now().year,
        }

    @app.teardown_appcontext
    def _remove_session(_exception=None):
        db.session.remove()

    def _error_response(code, message):
        if request.path.startswith("/api/"):
            return jsonify(error=message, status=code), code

        return (
            render_template("error.html", code=code, message=message),
            code,
        )

    @app.errorhandler(400)
    def bad_request(error):
        return _error_response(
            400, getattr(error, "description", None) or "Bad request."
        )

    @app.errorhandler(403)
    def forbidden(error):
        return _error_response(
            403,
            getattr(error, "description", None)
            or "You do not have permission to view this page.",
        )

    @app.errorhandler(404)
    def not_found(error):
        return _error_response(404, "The page you requested was not found.")

    @app.errorhandler(405)
    def method_not_allowed(error):
        return _error_response(
            405, "That action is not allowed on this address."
        )

    @app.errorhandler(413)
    def too_large(error):
        return _error_response(
            413, "The uploaded file is too large. The limit is 4 MB."
        )

    @app.errorhandler(503)
    def service_unavailable(error):
        return _error_response(
            503,
            getattr(error, "description", None)
            or "A required service is unavailable.",
        )

    @app.errorhandler(500)
    def server_error(error):
        db.session.rollback()
        app.logger.exception("Unhandled server error")
        return _error_response(
            500, "Something went wrong on our side. Please try again."
        )

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
