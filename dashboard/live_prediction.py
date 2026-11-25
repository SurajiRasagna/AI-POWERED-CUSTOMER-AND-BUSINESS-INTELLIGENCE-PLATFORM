"""
Live product-demand prediction helpers.

These functions wrap the models saved by
notebooks/step10_product_demand_forecasting.py so the dashboard can score
new/live demand data without retraining anything.

Model contract (must match step 10):
    features = time_index, month_number, lag_1, lag_2, lag_3, rolling_mean_3
    target   = quantity_sold (monthly units)
"""

from __future__ import annotations

import math

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "time_index",
    "month_number",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
]

REQUIRED_BATCH_COLUMNS = ["product_id", "month", "quantity_sold"]


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_models(model_path):
    """Load the pickled dict of {product_id: RandomForestRegressor}."""
    models = joblib.load(model_path)

    if not isinstance(models, dict) or not models:
        raise ValueError(
            "The product demand model file does not contain any models."
        )

    return models


def normalize_history(frame):
    """Validate and normalise monthly product-demand history."""
    required = {"product_id", "month", "quantity_sold"}

    if frame is None or frame.empty:
        raise ValueError("Monthly product demand history is empty.")

    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "History is missing columns: " + ", ".join(sorted(missing))
        )

    history = frame[["product_id", "month", "quantity_sold"]].copy()
    history["month"] = pd.to_datetime(history["month"], errors="coerce")
    history["quantity_sold"] = pd.to_numeric(
        history["quantity_sold"], errors="coerce"
    ).fillna(0.0)

    # Keep an optional time_index column so callers can align batch rows
    # with the model's training timeline.
    if "time_index" in frame.columns:
        history["time_index"] = pd.to_numeric(
            frame["time_index"], errors="coerce"
        )

    history = history.dropna(subset=["product_id", "month"])
    history = history.sort_values(["product_id", "month"])

    return history.reset_index(drop=True)


def product_options(history, products=None):
    """Per-product summary used to populate the dashboard selector."""
    summary = (
        history.groupby("product_id")
        .agg(
            months=("month", "nunique"),
            total_units=("quantity_sold", "sum"),
            last_month=("month", "max"),
            last_units=("quantity_sold", "last"),
        )
        .reset_index()
    )

    summary["category"] = "unknown"

    if products is not None and not products.empty:
        id_col = _find_column(products, ["product_id"])
        category_col = _find_column(
            products,
            ["product_category_name", "product_category", "category"],
        )

        if id_col and category_col:
            category_map = (
                products.dropna(subset=[id_col])
                .drop_duplicates(id_col)
                .set_index(id_col)[category_col]
            )
            summary["category"] = (
                summary["product_id"].map(category_map)
                .fillna("unknown")
                .astype(str)
            )

    summary["label"] = (
        summary["product_id"].str.slice(0, 10)
        + "…  |  "
        + summary["category"].str.slice(0, 24)
        + "  |  last "
        + summary["last_units"].round(0).astype(int).astype(str)
        + " units"
    )

    return summary.sort_values("total_units", ascending=False).reset_index(
        drop=True
    )


def get_product_history(history, product_id, limit=None):
    """Return one product's chronological monthly series."""
    series = history[history["product_id"] == product_id].copy()
    series = series.sort_values("month").reset_index(drop=True)

    if limit is not None and len(series) > limit:
        series = series.tail(limit).reset_index(drop=True)

    return series


# ---------------------------------------------------------------------------
# Feature construction and prediction
# ---------------------------------------------------------------------------

def build_feature_row(values, forecast_month, time_index=None):
    """
    Build a single model-ready row from the last three monthly values.

    values must be ordered oldest -> newest, i.e. [lag_3, lag_2, lag_1].
    """
    if len(values) < 3:
        raise ValueError("At least three monthly values are required.")

    lag_3, lag_2, lag_1 = values[-3], values[-2], values[-1]
    month = pd.Timestamp(forecast_month)

    if time_index is None:
        time_index = 0

    return pd.DataFrame([{
        "time_index": int(time_index),
        "month_number": int(month.month),
        "lag_1": float(lag_1),
        "lag_2": float(lag_2),
        "lag_3": float(lag_3),
        "rolling_mean_3": float(np.mean([lag_1, lag_2, lag_3])),
    }])[FEATURE_COLUMNS]


def predict_demand(models, product_id, history_values, forecast_month,
                   time_index=None):
    """Predict one month of demand for a single product."""
    if product_id not in models:
        raise KeyError(
            "No trained model is available for this product. "
            "Demand models were trained only for the top products."
        )

    row = build_feature_row(
        history_values,
        forecast_month,
        time_index=time_index,
    )

    prediction = float(models[product_id].predict(row)[0])

    return max(0.0, prediction)


def forecast_next_months(models, product_id, history_values, start_month,
                         horizon=3, time_index_start=None):
    """
    Recursively forecast the next `horizon` months.

    Each prediction is fed back as the most recent lag, matching the
    approach used when the models were trained.
    """
    if horizon < 1:
        raise ValueError("Horizon must be at least one month.")

    values = [float(v) for v in history_values]
    if len(values) < 3:
        raise ValueError("At least three monthly values are required.")

    start_month = pd.Timestamp(start_month) + pd.offsets.MonthBegin(0)

    base_index = (
        int(time_index_start) if time_index_start is not None else len(values)
    )

    rows = []

    for step in range(horizon):
        forecast_month = start_month + pd.offsets.MonthBegin(step)

        prediction = predict_demand(
            models,
            product_id,
            values[-3:],
            forecast_month,
            time_index=base_index + step,
        )

        rows.append({
            "forecast_month": forecast_month,
            "predicted_quantity": round(prediction),
            "predicted_quantity_raw": prediction,
        })

        values.append(prediction)

    return pd.DataFrame(rows)


def batch_forecast(models, frame, horizon=1, reference_history=None):
    """
    Score a batch of live records.

    Expected columns: product_id, month, quantity_sold.
    The latest three rows per product are used to predict the following
    month (or `horizon` months if greater than one).

    time_index alignment:
        1. If the file has a time_index column, its maximum is used.
        2. Otherwise, when reference_history is supplied, the product's
           known series length plus any elapsed months is used.
        3. Otherwise, the number of uploaded months is used.
    """
    if frame is None or frame.empty:
        raise ValueError("Uploaded file contains no rows.")

    missing = [c for c in REQUIRED_BATCH_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "Uploaded file is missing columns: " + ", ".join(missing)
        )

    data = normalize_history(frame)
    references = {}

    if reference_history is not None and not reference_history.empty:
        reference = normalize_history(reference_history)
        for product_id, group in reference.groupby("product_id"):
            references[product_id] = (
                int(group["month"].nunique()),
                group["month"].max(),
            )

    results = []

    for product_id, group in data.groupby("product_id"):
        group = group.sort_values("month")
        values = group["quantity_sold"].astype(float).tolist()

        base_row = {
            "product_id": product_id,
            "history_months": len(group),
            "last_month": group["month"].max(),
        }

        if len(values) < 3:
            results.append({
                **base_row,
                "status": "skipped: fewer than 3 months of history",
                "forecast_month": pd.NaT,
                "predicted_quantity": np.nan,
            })
            continue

        if product_id not in models:
            results.append({
                **base_row,
                "status": "skipped: no trained model",
                "forecast_month": pd.NaT,
                "predicted_quantity": np.nan,
            })
            continue

        start_index = _resolve_time_index(
            product_id,
            group,
            references,
        )

        try:
            forecast = forecast_next_months(
                models,
                product_id,
                values,
                group["month"].max() + pd.offsets.MonthBegin(1),
                horizon=horizon,
                time_index_start=start_index,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the dashboard
            results.append({
                **base_row,
                "status": f"error: {exc}",
                "forecast_month": pd.NaT,
                "predicted_quantity": np.nan,
            })
            continue

        for _, row in forecast.iterrows():
            results.append({
                **base_row,
                "status": "ok",
                "forecast_month": row["forecast_month"],
                "predicted_quantity": int(row["predicted_quantity"]),
            })

    return pd.DataFrame(results)


def _resolve_time_index(product_id, group, references):
    """Choose the next time_index for a batch product."""
    if "time_index" in group.columns:
        numeric = pd.to_numeric(group["time_index"], errors="coerce")
        if numeric.notna().any():
            return int(numeric.max()) + 1

    if product_id in references:
        known_months, known_last = references[product_id]
        last_uploaded = group["month"].max()
        elapsed = (
            (last_uploaded.year - known_last.year) * 12
            + (last_uploaded.month - known_last.month)
        )
        return known_months + max(0, int(elapsed))

    return len(group)


# ---------------------------------------------------------------------------
# Operational planning helpers
# ---------------------------------------------------------------------------

def demand_signal(prediction, history_values, window=3):
    """Classify the forecast against recent average demand."""
    recent = [float(v) for v in history_values[-window:]]
    if not recent:
        return "Unknown", 1.0

    average = float(np.mean(recent))
    if average <= 0:
        return ("High demand" if prediction > 0 else "No demand"), 1.0

    ratio = prediction / average

    if ratio >= 1.30:
        label = "High demand"
    elif ratio >= 0.80:
        label = "Normal demand"
    else:
        label = "Low demand"

    return label, ratio


def safety_stock(test_predictions, product_id, service_z=1.65):
    """
    Safety stock from the model's historical residual error.

    service_z = 1.65 approximates a 95% service level.
    """
    if test_predictions is None or test_predictions.empty:
        return 0.0, 0.0

    required = {"product_id", "actual_demand", "predicted_demand"}
    if not required.issubset(test_predictions.columns):
        return 0.0, 0.0

    rows = test_predictions[test_predictions["product_id"] == product_id]
    if rows.empty:
        return 0.0, 0.0

    residuals = (
        pd.to_numeric(rows["actual_demand"], errors="coerce")
        - pd.to_numeric(rows["predicted_demand"], errors="coerce")
    ).dropna()

    if residuals.empty:
        return 0.0, 0.0

    error_std = float(residuals.std(ddof=0))
    if not math.isfinite(error_std):
        return 0.0, 0.0

    return max(0.0, service_z * error_std), error_std


def _find_column(frame, possibilities):
    lookup = {str(c).lower().strip(): c for c in frame.columns}
    for name in possibilities:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None
