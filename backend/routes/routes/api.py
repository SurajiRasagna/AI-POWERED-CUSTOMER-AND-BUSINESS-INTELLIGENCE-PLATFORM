"""JSON API consumed by the web frontend."""

import math

import pandas as pd
from flask import Blueprint, abort, g, jsonify, request

import data_service
import ml_service
from database import db
from models import PredictionLog
from security import analyst_required, login_required

bp = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_body():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        abort(400, description="Expected a JSON object in the request body.")

    return payload


def _number(payload, field):
    try:
        value = float(payload.get(field))
    except (TypeError, ValueError):
        abort(400, description=f"'{field}' must be a number.")

    if not math.isfinite(value) or value < 0:
        abort(400, description=f"'{field}' must be zero or greater.")

    return value


def _month(value):
    try:
        stamp = pd.Timestamp(str(value or "").strip())
    except (TypeError, ValueError):
        abort(400, description="'forecast_month' must look like YYYY-MM.")

    if pd.isna(stamp):
        abort(400, description="'forecast_month' must look like YYYY-MM.")

    return stamp.to_period("M").to_timestamp()


def _product_row(product_id):
    options = ml_service.get_products()
    match = options[options["product_id"] == product_id]

    if match.empty:
        abort(
            400,
            description=(
                "Unknown product. Pick a product from the trained list."
            ),
        )

    row = match.iloc[0]

    return {
        "product_id": row["product_id"],
        "label": row["label"],
        "category": row["category"],
        "last_month": pd.Timestamp(row["last_month"]).strftime("%Y-%m"),
        "last_units": round(float(row["last_units"]), 2),
        "months": int(row["months"]),
    }


def _months_between(start, end):
    return (end.year - start.year) * 12 + (end.month - start.month)


# ---------------------------------------------------------------------------
# Analytics endpoints
# ---------------------------------------------------------------------------

@bp.get("/overview")
@login_required
def overview():
    return jsonify(data_service.overview_payload())


@bp.get("/sales")
@login_required
def sales():
    return jsonify(data_service.sales_payload())


@bp.get("/customers")
@login_required
def customers():
    return jsonify(data_service.customers_payload())


@bp.get("/demand")
@login_required
def demand():
    payload = data_service.demand_payload()

    try:
        payload["model"] = ml_service.model_info()
    except FileNotFoundError:
        payload["model"] = None

    return jsonify(payload)


@bp.get("/sentiment")
@login_required
def sentiment():
    return jsonify(data_service.sentiment_payload())


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

@bp.get("/products")
@login_required
def products():
    options = ml_service.get_products()
    rows = []

    for row in options.itertuples():
        rows.append(
            {
                "product_id": row.product_id,
                "label": row.label,
                "category": row.category,
                "months": int(row.months),
                "total_units": round(float(row.total_units), 2),
                "last_month": pd.Timestamp(row.last_month).strftime("%Y-%m"),
                "last_units": round(float(row.last_units), 2),
            }
        )

    return jsonify(
        products=rows,
        model=ml_service.model_info(),
    )


@bp.post("/predict/demand")
@analyst_required
def predict_demand():
    payload = _json_body()

    product_id = str(payload.get("product_id") or "").strip()
    if not product_id:
        abort(400, description="'product_id' is required.")

    lag_1 = _number(payload, "lag_1")
    lag_2 = _number(payload, "lag_2")
    lag_3 = _number(payload, "lag_3")

    forecast_month = _month(payload.get("forecast_month"))
    product = _product_row(product_id)

    last_month = pd.Timestamp(product["last_month"] + "-01")

    if forecast_month <= last_month:
        abort(
            400,
            description=(
                "Forecast month must be after the latest month in the "
                f"data ({product['last_month']})."
            ),
        )

    series = ml_service.get_product_history(product_id)

    default_time_index = len(series) + max(
        0, _months_between(last_month, forecast_month)
    )

    time_index = payload.get("time_index", default_time_index)

    try:
        time_index = int(time_index)
    except (TypeError, ValueError):
        abort(400, description="'time_index' must be a whole number.")

    if time_index < 0:
        abort(400, description="'time_index' cannot be negative.")

    result = ml_service.predict(
        product_id,
        lag_1,
        lag_2,
        lag_3,
        forecast_month,
        time_index=time_index,
    )

    forecast_rows = [
        {
            "forecast_month": row.forecast_month.strftime("%Y-%m"),
            "predicted_quantity": int(row.predicted_quantity),
        }
        for row in result["forecast"].itertuples()
    ]

    prediction = float(result["prediction"])
    safety = float(result["safety"])
    reorder_point = prediction + safety

    log = PredictionLog(
        product_id=product_id,
        forecast_month=forecast_month.date(),
        predicted_quantity=prediction,
        signal=result["signal"],
        safety_stock=safety,
        reorder_point=reorder_point,
        source="single",
        created_by=g.user.id,
    )
    db.session.add(log)
    db.session.commit()

    return jsonify(
        product=product,
        prediction=round(prediction, 2),
        forecast=forecast_rows,
        signal=result["signal"],
        ratio=round(float(result["ratio"]), 3),
        safety_stock=round(safety, 2),
        reorder_point=round(reorder_point, 2),
        error_std=round(float(result["error_std"]), 3),
        time_index=time_index,
    )


@bp.post("/predict/demand/batch")
@analyst_required
def predict_batch():
    uploaded = request.files.get("file")

    if uploaded is None or not uploaded.filename:
        abort(400, description="Upload a CSV file in the 'file' field.")

    if not uploaded.filename.lower().endswith(".csv"):
        abort(400, description="Only CSV files are supported.")

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        abort(400, description=f"Could not read the CSV: {exc}")

    if frame.empty:
        abort(400, description="The uploaded CSV has no rows.")

    if len(frame) > 5000:
        abort(400, description="The limit is 5,000 rows per upload.")

    try:
        result = ml_service.batch(frame)
    except ValueError as exc:
        abort(400, description=str(exc))

    rows = []

    for row in result.itertuples():
        rows.append(
            {
                "product_id": row.product_id,
                "status": row.status,
                "history_months": int(row.history_months),
                "last_month": (
                    row.last_month.strftime("%Y-%m")
                    if pd.notna(row.last_month)
                    else None
                ),
                "forecast_month": (
                    row.forecast_month.strftime("%Y-%m")
                    if pd.notna(row.forecast_month)
                    else None
                ),
                "predicted_quantity": (
                    int(row.predicted_quantity)
                    if pd.notna(row.predicted_quantity)
                    else None
                ),
            }
        )

    ok_rows = [row for row in rows if row["status"] == "ok"]
    total_units = sum(row["predicted_quantity"] or 0 for row in ok_rows)

    # Keep an audit trail, capped to avoid unbounded logging on big files.
    for row in ok_rows[:200]:
        safety, _ = ml_service.safety_for(row["product_id"])
        predicted = float(row["predicted_quantity"] or 0)

        db.session.add(
            PredictionLog(
                product_id=row["product_id"],
                forecast_month=pd.Timestamp(
                    row["forecast_month"] + "-01"
                ).date(),
                predicted_quantity=predicted,
                signal="batch",
                safety_stock=safety,
                reorder_point=predicted + safety,
                source="batch",
                created_by=g.user.id,
            )
        )

    db.session.commit()

    return jsonify(
        results=rows,
        summary={
            "scored": len(ok_rows),
            "skipped": len(rows) - len(ok_rows),
            "total_units": total_units,
        },
    )


@bp.get("/predictions")
@login_required
def predictions():
    try:
        limit = int(request.args.get("limit", 25))
    except ValueError:
        limit = 25

    limit = min(max(limit, 1), 200)

    query = PredictionLog.query

    if not g.user.is_admin:
        query = query.filter_by(created_by=g.user.id)

    logs = (
        query.order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )

    return jsonify(predictions=[entry.to_dict() for entry in logs])


@bp.get("/ml/info")
@login_required
def model_info():
    try:
        return jsonify(ml_service.model_info())
    except FileNotFoundError as exc:
        abort(503, description=str(exc))
