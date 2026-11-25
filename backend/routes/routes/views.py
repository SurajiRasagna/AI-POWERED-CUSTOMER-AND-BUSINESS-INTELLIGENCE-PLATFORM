"""HTML page routes."""

from datetime import datetime, timezone

from flask import Blueprint, jsonify, redirect, render_template, url_for

from security import login_required

bp = Blueprint("views", __name__)


def _page(template, title, subtitle, **context):
    return render_template(
        template,
        page_title=title,
        page_subtitle=subtitle,
        **context,
    )


@bp.get("/health")
def health():
    return jsonify(
        status="ok",
        app="AI Customer & Business Intelligence",
        time=datetime.now(timezone.utc).isoformat(),
    )


@bp.get("/")
@login_required
def overview():
    return _page(
        "overview.html",
        "Business Overview",
        "A live snapshot of revenue, customers and demand across the store.",
    )


@bp.get("/sales")
@login_required
def sales():
    return _page(
        "sales.html",
        "Sales Analytics",
        "Revenue trends, top categories, payment behaviour and forecasts.",
    )


@bp.get("/customers")
@login_required
def customers():
    return _page(
        "customers.html",
        "Customer Intelligence",
        "Segments, lifetime value and churn risk across your customer base.",
    )


@bp.get("/demand")
@login_required
def demand():
    return _page(
        "demand.html",
        "Product Demand",
        "Historical demand and machine-learning forecasts for top products.",
    )


@bp.get("/sentiment")
@login_required
def sentiment():
    return _page(
        "sentiment.html",
        "Customer Sentiment",
        "What customers say, classified from review text and ratings.",
    )


@bp.get("/live")
@login_required
def live():
    return _page(
        "live.html",
        "Live Demand Prediction",
        "Score new demand figures instantly with the trained models.",
    )


@bp.get("/admin")
@login_required
def admin_redirect():
    return redirect(url_for("admin.users"))
