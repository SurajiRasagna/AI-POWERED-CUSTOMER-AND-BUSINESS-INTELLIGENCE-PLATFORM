"""Admin-only user management."""

from flask import (
    Blueprint,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

from database import db
from models import ROLES, User
from security import admin_required

bp = Blueprint("admin", __name__, url_prefix="/admin")


def _active_admin_count():
    return User.query.filter_by(role="admin", is_active=True).count()


@bp.get("/users")
@admin_required
def users():
    accounts = User.query.order_by(User.created_at.asc()).all()

    stats = {
        "total": len(accounts),
        "active": sum(1 for u in accounts if u.is_active),
        "admins": sum(
            1 for u in accounts if u.role == "admin" and u.is_active
        ),
    }

    return render_template(
        "admin_users.html",
        page_title="User Management",
        page_subtitle="Control who can access the platform and what they can do.",
        accounts=accounts,
        roles=ROLES,
        stats=stats,
    )


@bp.post("/users/<int:user_id>/role")
@admin_required
def update_role(user_id):
    user = db.get_or_404(User, user_id)
    role = request.form.get("role", "")

    if role not in ROLES:
        flash("Unknown role.", "error")
        return redirect(url_for("admin.users"))

    if (
        user.role == "admin"
        and role != "admin"
        and user.is_active
        and _active_admin_count() <= 1
    ):
        flash("At least one active admin must remain.", "error")
        return redirect(url_for("admin.users"))

    user.role = role
    db.session.commit()
    flash(f"{user.name} is now a {role}.", "success")
    return redirect(url_for("admin.users"))


@bp.post("/users/<int:user_id>/toggle")
@admin_required
def toggle_user(user_id):
    user = db.get_or_404(User, user_id)

    if user.id == g.user.id:
        flash("You cannot deactivate your own account.", "error")
        return redirect(url_for("admin.users"))

    if user.is_active and user.is_admin and _active_admin_count() <= 1:
        flash("At least one active admin must remain.", "error")
        return redirect(url_for("admin.users"))

    user.is_active = not user.is_active
    db.session.commit()

    state = "activated" if user.is_active else "deactivated"
    flash(f"{user.name} has been {state}.", "success")
    return redirect(url_for("admin.users"))
