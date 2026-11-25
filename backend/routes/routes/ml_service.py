"""Machine-learning service layer for the web app.

Loads the product-demand models once per file version and exposes simple
functions that the JSON API can call. Reuses the exact prediction logic from
dashboard/live_prediction.py so the web app and Streamlit never diverge.
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DIR = ROOT / "dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import pandas as pd  # noqa: E402

import live_prediction as live  # noqa: E402
from config import DATA_DIR, MODELS_DIR  # noqa: E402

MODEL_FILE = MODELS_DIR / "product_demand_forecasting_models.pkl"
DEMAND_HISTORY_FILE = (
    DATA_DIR
    / "product_demand_forecasting_outputs"
    / "monthly_product_demand.csv"
)
PRODUCTS_FILE = DATA_DIR / "products_cleaned.csv"
TEST_PREDICTIONS_FILE = (
    DATA_DIR
    / "product_demand_forecasting_outputs"
    / "product_demand_test_predictions.csv"
)

_cache = {}


def _cached(key, mtime, loader):
    cache_key = (key, mtime)

    if cache_key not in _cache:
        for existing in list(_cache):
            if existing[0] == key and existing[1] != mtime:
                del _cache[existing]
        _cache[cache_key] = loader()

    return _cache[cache_key]


def get_models():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Demand model file is missing. Run step10 product demand "
            "forecasting first."
        )

    return _cached(
        "models",
        MODEL_FILE.stat().st_mtime,
        lambda: live.load_models(str(MODEL_FILE)),
    )


def get_history():
    if not DEMAND_HISTORY_FILE.exists():
        raise FileNotFoundError("Monthly product demand history is missing.")

    return _cached(
        "history",
        DEMAND_HISTORY_FILE.stat().st_mtime,
        lambda: live.normalize_history(pd.read_csv(DEMAND_HISTORY_FILE)),
    ).copy()


def get_products():
    def load():
        products = (
            pd.read_csv(PRODUCTS_FILE) if PRODUCTS_FILE.exists() else None
        )
        return live.product_options(get_history(), products)

    return _cached(
        "products",
        DEMAND_HISTORY_FILE.stat().st_mtime,
        load,
    ).copy()


def get_product_history(product_id):
    return live.get_product_history(get_history(), product_id)


def get_test_predictions():
    if not TEST_PREDICTIONS_FILE.exists():
        return pd.DataFrame()

    return _cached(
        "test_predictions",
        TEST_PREDICTIONS_FILE.stat().st_mtime,
        lambda: pd.read_csv(TEST_PREDICTIONS_FILE),
    ).copy()


def safety_for(product_id):
    """Safety stock and residual error for one product."""
    return live.safety_stock(get_test_predictions(), product_id)


def model_info():
    models = get_models()
    history = get_history()
    trained_at = datetime.fromtimestamp(MODEL_FILE.stat().st_mtime)

    return {
        "model_name": "RandomForestRegressor",
        "product_count": len(models),
        "trained_at": trained_at.strftime("%d %b %Y %H:%M"),
        "data_through": history["month"].max().strftime("%b %Y"),
        "features": list(live.FEATURE_COLUMNS),
    }


def predict(product_id, lag_1, lag_2, lag_3, forecast_month, time_index=None):
    models = get_models()

    prediction = live.predict_demand(
        models,
        product_id,
        [lag_3, lag_2, lag_1],
        forecast_month,
        time_index=time_index,
    )

    forecast = live.forecast_next_months(
        models,
        product_id,
        [lag_3, lag_2, lag_1],
        forecast_month,
        horizon=3,
        time_index_start=time_index,
    )

    signal, ratio = live.demand_signal(
        prediction,
        [lag_3, lag_2, lag_1],
    )

    safety, error_std = live.safety_stock(
        get_test_predictions(),
        product_id,
    )

    return {
        "prediction": prediction,
        "forecast": forecast,
        "signal": signal,
        "ratio": ratio,
        "safety": safety,
        "error_std": error_std,
    }


def batch(frame):
    return live.batch_forecast(
        get_models(),
        frame,
        horizon=1,
        reference_history=get_history(),
    )
