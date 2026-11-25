
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = r"C:\Users\suraj\OneDrive\AI_Customer_BI_Project"

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

OUTPUT_DIR = os.path.join(
    DATA_DIR, "product_demand_forecasting_outputs"
)

CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Step 10: Product Demand Forecasting Started")

# ==================================================
# 2. LOAD CLEANED DATA
# ==================================================

orders = pd.read_csv(
    os.path.join(DATA_DIR, "orders_cleaned.csv")
)

order_items = pd.read_csv(
    os.path.join(DATA_DIR, "order_items_cleaned.csv")
)

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

orders = orders.dropna(
    subset=["order_id", "order_purchase_timestamp"]
)

# Exclude canceled and unavailable orders
orders = orders[
    ~orders["order_status"].astype(str).str.lower().isin(
        ["canceled", "unavailable"]
    )
]

order_items = order_items.dropna(
    subset=["order_id", "product_id"]
)

print("Orders loaded:", len(orders))
print("Order items loaded:", len(order_items))

# ==================================================
# 3. MERGE ORDERS AND ORDER ITEMS
# ==================================================

data = pd.merge(
    order_items,
    orders[[
        "order_id",
        "order_purchase_timestamp"
    ]],
    on="order_id",
    how="inner"
)

data["month"] = (
    data["order_purchase_timestamp"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

# Each order-item record represents one unit
monthly_demand = (
    data.groupby(["product_id", "month"])
    .size()
    .reset_index(name="quantity_sold")
)

print("Monthly demand calculated.")

# ==================================================
# 4. SELECT TOP 10 PRODUCTS
# ==================================================

top_products = (
    monthly_demand.groupby("product_id")["quantity_sold"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

monthly_demand = monthly_demand[
    monthly_demand["product_id"].isin(top_products)
].copy()

# Create complete monthly timeline
all_months = pd.date_range(
    start=monthly_demand["month"].min(),
    end=monthly_demand["month"].max(),
    freq="MS"
)

product_panels = []

for product in top_products:

    product_data = (
        monthly_demand[
            monthly_demand["product_id"] == product
        ]
        .set_index("month")["quantity_sold"]
        .reindex(all_months, fill_value=0)
    )

    panel = pd.DataFrame({
        "product_id": product,
        "month": all_months,
        "quantity_sold": product_data.values
    })

    product_panels.append(panel)

monthly_panel = pd.concat(
    product_panels,
    ignore_index=True
)

monthly_panel.to_csv(
    os.path.join(OUTPUT_DIR, "monthly_product_demand.csv"),
    index=False
)

print("Top 10 product demand data prepared.")

# ==================================================
# 5. CREATE MACHINE LEARNING FEATURES
# ==================================================

def create_features(product_data):

    product_data = product_data.sort_values(
        "month"
    ).copy()

    product_data["time_index"] = range(
        len(product_data)
    )

    product_data["month_number"] = (
        product_data["month"].dt.month
    )

    product_data["lag_1"] = (
        product_data["quantity_sold"].shift(1)
    )

    product_data["lag_2"] = (
        product_data["quantity_sold"].shift(2)
    )

    product_data["lag_3"] = (
        product_data["quantity_sold"].shift(3)
    )

    product_data["rolling_mean_3"] = (
        product_data["quantity_sold"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    return product_data.dropna().copy()


FEATURES = [
    "time_index",
    "month_number",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3"
]

# ==================================================
# 6. TRAIN AND EVALUATE MODEL
# ==================================================

product_models = {}
evaluation_results = []
test_results = []
forecast_results = []

for product in top_products:

    product_data = monthly_panel[
        monthly_panel["product_id"] == product
    ].copy()

    featured_data = create_features(product_data)

    if len(featured_data) < 8:
        print(
            f"Skipping product {product}: insufficient data."
        )
        continue

    # Last 3 months used for testing
    train_data = featured_data.iloc[:-3]
    test_data = featured_data.iloc[-3:]

    X_train = train_data[FEATURES]
    y_train = train_data["quantity_sold"]

    X_test = test_data[FEATURES]
    y_test = test_data["quantity_sold"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = np.maximum(
        model.predict(X_test),
        0
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    evaluation_results.append({
        "product_id": product,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })

    for actual, predicted, month in zip(
        y_test,
        predictions,
        test_data["month"]
    ):

        test_results.append({
            "product_id": product,
            "month": month,
            "actual_demand": actual,
            "predicted_demand": round(predicted, 2)
        })

    # ==================================================
    # 7. TRAIN FINAL MODEL USING ALL HISTORY
    # ==================================================

    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    final_model.fit(
        featured_data[FEATURES],
        featured_data["quantity_sold"]
    )

    product_models[product] = final_model

    # ==================================================
    # 8. FORECAST NEXT 3 MONTHS
    # ==================================================

    history = (
        product_data["quantity_sold"]
        .astype(float)
        .tolist()
    )

    last_month = product_data["month"].max()

    for step in range(1, 4):

        future_month = (
            last_month + pd.offsets.MonthBegin(step)
        )

        feature_row = pd.DataFrame([{
            "time_index": len(history),
            "month_number": future_month.month,
            "lag_1": history[-1],
            "lag_2": history[-2],
            "lag_3": history[-3],
            "rolling_mean_3": np.mean(history[-3:])
        }])

        predicted_demand = max(
            0,
            float(
                final_model.predict(
                    feature_row[FEATURES]
                )[0]
            )
        )

        predicted_demand = round(
            predicted_demand
        )

        forecast_results.append({
            "product_id": product,
            "forecast_month": future_month,
            "predicted_quantity": predicted_demand
        })

        history.append(predicted_demand)

# ==================================================
# 9. SAVE CSV REPORTS
# ==================================================

evaluation_df = pd.DataFrame(
    evaluation_results
)

test_df = pd.DataFrame(
    test_results
)

forecast_df = pd.DataFrame(
    forecast_results
)

evaluation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "product_demand_model_evaluation.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "product_demand_test_predictions.csv"
    ),
    index=False
)

forecast_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "product_demand_forecast_next_3_months.csv"
    ),
    index=False
)

# ==================================================
# 10. SAVE TRAINED MODELS
# ==================================================

joblib.dump(
    product_models,
    os.path.join(
        MODEL_DIR,
        "product_demand_forecasting_models.pkl"
    )
)

print("CSV reports and models saved.")

# ==================================================
# 11. HISTORICAL DEMAND CHART
# ==================================================

historical_pivot = monthly_panel.pivot(
    index="month",
    columns="product_id",
    values="quantity_sold"
)

plt.figure(figsize=(14, 7))

historical_pivot.plot(
    ax=plt.gca()
)

plt.title(
    "Historical Monthly Demand - Top 10 Products"
)

plt.xlabel("Month")
plt.ylabel("Quantity Sold")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "historical_product_demand.png"
    )
)

plt.close()

# ==================================================
# 12. FUTURE DEMAND FORECAST CHART
# ==================================================

if not forecast_df.empty:

    forecast_pivot = forecast_df.pivot(
        index="forecast_month",
        columns="product_id",
        values="predicted_quantity"
    )

    plt.figure(figsize=(14, 7))

    forecast_pivot.plot(
        kind="bar",
        ax=plt.gca()
    )

    plt.title(
        "Product Demand Forecast - Next 3 Months"
    )

    plt.xlabel("Forecast Month")
    plt.ylabel("Predicted Quantity")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "future_product_demand.png"
        )
    )

    plt.close()

# ==================================================
# 13. DISPLAY FINAL RESULTS
# ==================================================

print("\n====================================")
print("PRODUCT DEMAND FORECASTING COMPLETED")
print("====================================")

print("\nModel Evaluation:")
print(evaluation_df)

print("\nNext 3 Months Product Demand:")
print(forecast_df)

print("\nOutput Folder:")
print(OUTPUT_DIR)

print("\nModel Saved:")
print(
    os.path.join(
        MODEL_DIR,
        "product_demand_forecasting_models.pkl"
    )
)
print("BASE DIR:", BASE_DIR)
print("DATA DIR:", DATA_DIR)
print("FILES:", os.listdir(DATA_DIR))