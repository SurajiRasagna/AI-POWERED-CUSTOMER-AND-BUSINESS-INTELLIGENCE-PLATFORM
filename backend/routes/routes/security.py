"""Session authentication, CSRF protection and role guards."""

import secrets
from functools import wraps

from flask import abort, g, jsonify, redirect, request, session, url_for

from database import db
from models import User

SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


def _wants_json():
    return request.path.startswith("/api/") or request.is_json


def get_csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def validate_csrf():
    """Reject state-changing requests without a matching CSRF token."""
    if request.method in SAFE_METHODS:
        return None

    sent = request.headers.get("X-CSRF-Token") or request.form.get(
        "csrf_token"
    )
    expected = session.get("_csrf_token", "")

    if not sent or not expected or not secrets.compare_digest(
        str(sent), str(expected)
    ):
        message = "Invalid or missing security token. Refresh and try again."

        if _wants_json():
            return jsonify(error=message), 400

        abort(400, description=message)

    return None


def load_current_user():
    """Attach the logged-in user to the request context."""
    user_id = session.get("user_id")
    user = db.session.get(User, user_id) if user_id else None

    if user is not None and not user.is_active:
        session.clear()
        user = None

    g.user = user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.get("user") is None:
            if _wants_json():
                return jsonify(error="Authentication required."), 401

            return redirect(
                url_for(
                    "auth.login",
                    next=request.full_path.rstrip("?"),
                )
            )

        return view(*args, **kwargs)

    return wrapped


def role_required(*roles):
    """Require one of the given roles in addition to being logged in."""

    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = g.get("user")

            if user is None:
                if _wants_json():
                    return jsonify(error="Authentication required."), 401

                return redirect(url_for("auth.login"))

            if user.role not in roles:
                if _wants_json():
                    return (
                        jsonify(error="You do not have permission for this action."),
                        403,
                    )

                abort(403, description="You do not have permission for this page.")

            return view(*args, **kwargs)

        return wrapped

    return decorator


admin_required = role_required("admin")
analyst_required = role_required("admin", "analyst")


def safe_next_url(candidate):
    """Only allow same-site relative redirects."""
    if not candidate:
        return None
    if not candidate.startswith("/") or candidate.startswith("//"):
        return None
    return candidate
