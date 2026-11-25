"""Authentication routes: register, login and logout."""

import re

from flask import (
    Blueprint,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from database import db
from models import User, utcnow
from security import safe_next_url

bp = Blueprint("auth", __name__)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _clean_email(value):
    return (value or "").strip().lower()


def _redirect_after_login():
    return safe_next_url(request.args.get("next") or request.form.get("next"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.get("user") is not None:
        return redirect(url_for("views.overview"))

    if request.method == "POST":
        email = _clean_email(request.form.get("email"))
        password = request.form.get("password") or ""

        user = User.query.filter_by(email=email).first()

        if user is None or not user.check_password(password):
            flash("Incorrect email or password.", "error")
            return render_template("login.html", email=email), 401

        if not user.is_active:
            flash("This account has been deactivated. Contact an admin.", "error")
            return render_template("login.html", email=email), 403

        session.clear()
        session["user_id"] = user.id
        session.permanent = True

        user.last_login_at = utcnow()
        db.session.commit()

        flash(f"Welcome back, {user.name.split()[0]}.", "success")
        return redirect(_redirect_after_login() or url_for("views.overview"))

    return render_template("login.html", email="")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.get("user") is not None:
        return redirect(url_for("views.overview"))

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = _clean_email(request.form.get("email"))
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""

        errors = []

        if len(name) < 2:
            errors.append("Enter your full name.")

        if not EMAIL_RE.match(email):
            errors.append("Enter a valid email address.")

        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")

        if password != confirm:
            errors.append("Passwords do not match.")

        if User.query.filter_by(email=email).first() is not None:
            errors.append("An account with this email already exists.")

        if errors:
            for message in errors:
                flash(message, "error")
            return (
                render_template("register.html", name=name, email=email),
                400,
            )

        is_first_user = User.query.count() == 0
        role = "admin" if is_first_user else "analyst"

        user = User(name=name, email=email, role=role)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        session.clear()
        session["user_id"] = user.id
        session.permanent = True

        if is_first_user:
            flash(
                "Admin account created. You can manage users from the Users page.",
                "success",
            )
        else:
            flash("Account created. Welcome aboard.", "success")

        return redirect(url_for("views.overview"))

    return render_template("register.html", name="", email="")


@bp.post("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("auth.login"))
