"""Reads the processed analytics outputs and shapes them for the JSON API.

Each file is cached against its modification time, so re-running any notebook
refreshes the web app without a server restart.
"""

import json

import pandas as pd

from config import DATA_DIR

_cache = {}


def _load(relative_path, columns=None):
    path = DATA_DIR / relative_path

    if not path.exists():
        return pd.DataFrame()

    mtime = path.stat().st_mtime
    cache_key = (relative_path, tuple(columns or ()), mtime)

    if cache_key not in _cache:
        for existing in list(_cache):
            if (
                existing[0] == relative_path
                and existing[1] == tuple(columns or ())
                and existing[2] != mtime
            ):
                del _cache[existing]
        _cache[cache_key] = pd.read_csv(path, usecols=columns)

    return _cache[cache_key].copy()


def records(relative_path, columns=None):
    frame = _load(relative_path, columns)
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def metric_map(relative_path):
    frame = _load(relative_path)
    if frame.empty or "Metric" not in frame.columns:
        return {}

    return {
        str(metric).strip().lower(): value
        for metric, value in zip(frame["Metric"], frame["Value"])
    }


def _num(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _months(relative_path, column):
    frame = _load(relative_path, [column])
    if frame.empty:
        return []

    months = pd.to_datetime(frame[column], errors="coerce").dropna()
    return sorted(months.dt.strftime("%Y-%m").unique().tolist())


def overview_payload():
    sales = metric_map("sales_revenue_outputs/overall_sales_summary.csv")
    customers = metric_map(
        "customer_analytics_outputs/customer_summary.csv"
    )

    total_sales = _num(sales.get("total product sales"))
    freight = _num(sales.get("total freight value"))
    total_revenue = _num(
        sales.get("total revenue"), total_sales + freight
    )
    orders = _num(sales.get("total delivered orders"))
    unique_customers = _num(customers.get("unique customers"))

    return {
        "kpis": {
            "revenue": total_revenue,
            "product_sales": total_sales,
            "orders": int(orders),
            "customers": int(unique_customers),
            "average_order_value": (
                total_revenue / orders if orders else 0.0
            ),
        },
        "monthly_sales": records(
            "sales_revenue_outputs/monthly_sales_revenue.csv"
        ),
        "top_categories": records(
            "sales_revenue_outputs/top_10_categories.csv"
        ),
        "sentiment": records(
            "sentiment_analysis_outputs/sentiment_summary.csv"
        ),
        "demand_forecast": records(
            "product_demand_forecasting_outputs/"
            "product_demand_forecast_next_3_months.csv"
        ),
    }


def sales_payload():
    sales = metric_map("sales_revenue_outputs/overall_sales_summary.csv")

    return {
        "kpis": {
            "total_sales": _num(sales.get("total product sales")),
            "freight": _num(sales.get("total freight value")),
            "orders": int(_num(sales.get("total delivered orders"))),
        },
        "monthly": records(
            "sales_revenue_outputs/monthly_sales_revenue.csv"
        ),
        "top_categories": records(
            "sales_revenue_outputs/top_10_categories.csv"
        ),
        "top_products": records(
            "sales_revenue_outputs/top_10_products.csv"
        ),
        "states": records("sales_revenue_outputs/state_wise_sales.csv"),
        "payments": records(
            "sales_revenue_outputs/payment_method_analysis.csv"
        ),
        "forecast": records(
            "sales_forecasting_outputs/next_3_months_sales_forecast.csv"
        ),
        "forecast_models": records(
            "sales_forecasting_outputs/forecast_model_comparison.csv"
        ),
        "historical_months": _months(
            "sales_forecasting_outputs/historical_monthly_sales.csv",
            "month",
        ),
    }


def customers_payload():
    summary = metric_map(
        "customer_analytics_outputs/customer_summary.csv"
    )
    clv_segments = records(
        "customer_lifetime_value_outputs/customer_value_segment_summary.csv"
    )

    total_clv_revenue = sum(
        _num(row.get("total_revenue")) for row in clv_segments
    )
    total_clv_customers = sum(
        _num(row.get("customer_count")) for row in clv_segments
    )

    churn = _load(
        "churn_prediction_outputs/customer_churn_predictions.csv",
        ["risk_level"],
    )

    churn_risk = []
    if not churn.empty and "risk_level" in churn.columns:
        counts = churn["risk_level"].value_counts()
        for label in ["Low Risk", "Medium Risk", "High Risk"]:
            churn_risk.append(
                {"risk_level": label, "count": int(counts.get(label, 0))}
            )

    return {
        "kpis": {
            "customers": int(_num(summary.get("unique customers"))),
            "repeat_customers": int(
                _num(summary.get("repeat customers"))
            ),
            "average_clv": (
                total_clv_revenue / total_clv_customers
                if total_clv_customers
                else 0.0
            ),
            "high_value_customers": next(
                (
                    int(_num(row.get("customer_count")))
                    for row in clv_segments
                    if str(row.get("customer_segment")) == "High Value"
                ),
                0,
            ),
        },
        "rfm_segments": records(
            "customer_analytics_outputs/customer_segment_summary.csv"
        ),
        "clv_segments": clv_segments,
        "top_customers": records(
            "customer_analytics_outputs/top_10_customers.csv"
        ),
        "top_clv_customers": records(
            "customer_lifetime_value_outputs/"
            "top_10_high_value_customers.csv"
        ),
        "churn_risk": churn_risk,
        "churn_models": records(
            "churn_prediction_outputs/model_comparison.csv"
        ),
    }


def demand_payload():
    frame = _load(
        "product_demand_forecasting_outputs/monthly_product_demand.csv"
    )
    history = []

    if not frame.empty:
        dates = pd.to_datetime(frame["month"], errors="coerce")
        history = [
            {
                "product_id": row.product_id,
                "month": month.strftime("%Y-%m") if pd.notna(month) else "",
                "quantity_sold": float(row.quantity_sold),
            }
            for row, month in zip(frame.itertuples(), dates)
        ]

    forecast = _load(
        "product_demand_forecasting_outputs/"
        "product_demand_forecast_next_3_months.csv"
    )
    forecast_records = []

    if not forecast.empty:
        months = pd.to_datetime(
            forecast["forecast_month"], errors="coerce"
        )
        forecast_records = [
            {
                "product_id": row.product_id,
                "forecast_month": (
                    month.strftime("%Y-%m") if pd.notna(month) else ""
                ),
                "predicted_quantity": float(row.predicted_quantity),
            }
            for row, month in zip(forecast.itertuples(), months)
        ]

    return {
        "history": history,
        "forecast": forecast_records,
        "evaluation": records(
            "product_demand_forecasting_outputs/"
            "product_demand_model_evaluation.csv"
        ),
    }


def sentiment_payload():
    distribution = records(
        "sentiment_analysis_outputs/sentiment_summary.csv"
    )
    reviews = _load(
        "sentiment_analysis_outputs/customer_sentiment_analysis.csv",
        [
            "review_score",
            "predicted_sentiment",
            "review_comment_message",
            "review_creation_date",
        ],
    )

    ratings = []
    recent = []
    kpis = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}

    if not reviews.empty:
        counts = (
            reviews["predicted_sentiment"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
        )

        kpis = {
            "positive": int(counts.get("Positive", 0)),
            "neutral": int(counts.get("Neutral", 0)),
            "negative": int(counts.get("Negative", 0)),
            "total": int(counts.sum()),
        }

        score_counts = (
            pd.to_numeric(reviews["review_score"], errors="coerce")
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        ratings = [
            {"review_score": int(score), "reviews": int(count)}
            for score, count in score_counts.items()
        ]

        with_text = reviews[
            reviews["review_comment_message"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.len()
            > 0
        ].tail(12)

        for row in with_text.iloc[::-1].itertuples():
            message = str(row.review_comment_message).strip()
            recent.append(
                {
                    "review_score": int(row.review_score)
                    if pd.notna(row.review_score)
                    else None,
                    "sentiment": str(row.predicted_sentiment),
                    "message": message[:240],
                    "date": str(row.review_creation_date)[:10],
                }
            )

    return {
        "kpis": kpis,
        "distribution": distribution,
        "ratings": ratings,
        "recent": recent,
        "evaluation": records(
            "sentiment_analysis_outputs/sentiment_model_evaluation.csv"
        ),
    }
