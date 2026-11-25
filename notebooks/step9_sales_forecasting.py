
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------------
# 1. PROJECT PATHS
# -------------------------------------------------

DATA_PATH = Path("data/processed")
OUTPUT_PATH = DATA_PATH / "sales_forecasting_outputs"
CHART_PATH = OUTPUT_PATH / "charts"
MODEL_PATH = Path("models")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CHART_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 2. LOAD CLEANED DATASETS
# -------------------------------------------------

orders = pd.read_csv(DATA_PATH / "orders_cleaned.csv")
order_items = pd.read_csv(DATA_PATH / "order_items_cleaned.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

order_items["price"] = pd.to_numeric(
    order_items["price"], errors="coerce"
).fillna(0)

# -------------------------------------------------
# 3. SELECT DELIVERED ORDERS
# -------------------------------------------------

delivered_orders = orders[
    (orders["order_status"] == "delivered") &
    (orders["order_purchase_timestamp"].notna())
].copy()

sales_data = delivered_orders.merge(
    order_items[["order_id", "price"]],
    on="order_id",
    how="inner"
)

# -------------------------------------------------
# 4. CALCULATE MONTHLY SALES
# -------------------------------------------------

sales_data["month"] = (
    sales_data["order_purchase_timestamp"].dt.to_period("M")
)

monthly_sales = (
    sales_data.groupby("month")["price"]
    .sum()
    .reset_index(name="sales")
)

# Fill missing calendar months with zero sales.
all_months = pd.period_range(
    start=monthly_sales["month"].min(),
    end=monthly_sales["month"].max(),
    freq="M"
)

monthly_sales = (
    monthly_sales.set_index("month")
    .reindex(all_months, fill_value=0)
    .rename_axis("month")
    .reset_index()
)

monthly_sales["month"] = monthly_sales["month"].astype(str)

monthly_sales.to_csv(
    OUTPUT_PATH / "historical_monthly_sales.csv",
    index=False
)

print("\n========== HISTORICAL SALES SUMMARY ==========")
print("Number of months:", len(monthly_sales))
print("Total product sales (BRL):", round(monthly_sales["sales"].sum(), 2))
print(monthly_sales.tail(10).to_string(index=False))

# -------------------------------------------------
# 5. CREATE TIME-SERIES FEATURES
# -------------------------------------------------

forecast_data = monthly_sales.copy()

forecast_data["month_date"] = pd.to_datetime(
    forecast_data["month"] + "-01"
)

forecast_data["month_number"] = (
    forecast_data["month_date"].dt.month
)

forecast_data["time_index"] = np.arange(
    len(forecast_data)
)

# Previous month sales
forecast_data["lag_1"] = forecast_data["sales"].shift(1)

# Sales from two and three months earlier
forecast_data["lag_2"] = forecast_data["sales"].shift(2)
forecast_data["lag_3"] = forecast_data["sales"].shift(3)

# Average sales of the previous three months
forecast_data["rolling_mean_3"] = (
    forecast_data["sales"].shift(1).rolling(3).mean()
)

forecast_data = forecast_data.dropna().reset_index(drop=True)

feature_columns = [
    "month_number",
    "time_index",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3"
]

X = forecast_data[feature_columns]
y = forecast_data["sales"]

if len(forecast_data) < 12:
    raise ValueError(
        "Not enough monthly history for this forecasting exercise. "
        "At least 15 calendar months are recommended."
    )

# -------------------------------------------------
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# -------------------------------------------------

# Keep the latest 6 available feature rows for testing.
test_size = 6

X_train = X.iloc[:-test_size]
X_test = X.iloc[-test_size:]

y_train = y.iloc[:-test_size]
y_test = y.iloc[-test_size:]

print("\nTraining months:", len(X_train))
print("Testing months:", len(X_test))

# -------------------------------------------------
# 7. TRAIN FORECASTING MODELS
# -------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42
    )
}

results = []
trained_models = {}

# Naive forecast: next month's sales equal the previous month.
naive_predictions = X_test["lag_1"].to_numpy()

naive_mae = mean_absolute_error(y_test, naive_predictions)
naive_rmse = np.sqrt(
    mean_squared_error(y_test, naive_predictions)
)

results.append({
    "Model": "Naive Baseline",
    "MAE": naive_mae,
    "RMSE": naive_rmse
})

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse
    })

    trained_models[model_name] = model

    print(f"{model_name} training completed.")

# -------------------------------------------------
# 8. COMPARE MODEL PERFORMANCE
# -------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH / "forecast_model_comparison.csv",
    index=False
)

print("\n========== FORECAST MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Select the model with the lowest RMSE.
best_model_name = (
    results_df.sort_values("RMSE").iloc[0]["Model"]
)

if best_model_name == "Naive Baseline":
    print("\nNaive Baseline performed best.")
    print("A trained machine-learning model will still be saved for later use.")
    best_model_name = "Linear Regression"

best_model = trained_models[best_model_name]

print("\nSelected machine learning model:", best_model_name)

# -------------------------------------------------
# 9. SAVE TEST PREDICTIONS
# -------------------------------------------------

test_predictions = best_model.predict(X_test)

test_results = pd.DataFrame({
    "month": forecast_data.loc[X_test.index, "month"].values,
    "actual_sales": y_test.values,
    "predicted_sales": test_predictions
})

test_results.to_csv(
    OUTPUT_PATH / "sales_test_predictions.csv",
    index=False
)

# -------------------------------------------------
# 10. FORECAST THE NEXT THREE MONTHS
# -------------------------------------------------

sales_history = monthly_sales["sales"].tolist()

last_month = pd.Period(
    monthly_sales["month"].iloc[-1],
    freq="M"
)

future_results = []

for step in range(1, 4):

    next_month = last_month + step

    lag_1 = sales_history[-1]
    lag_2 = sales_history[-2]
    lag_3 = sales_history[-3]

    rolling_mean_3 = np.mean(sales_history[-3:])

    next_time_index = len(monthly_sales) + step - 1

    future_features = pd.DataFrame([{
        "month_number": next_month.month,
        "time_index": next_time_index,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "rolling_mean_3": rolling_mean_3
    }])

    predicted_sales = float(
        best_model.predict(future_features[feature_columns])[0]
    )

    # Sales cannot be negative.
    predicted_sales = max(0, predicted_sales)

    future_results.append({
        "month": str(next_month),
        "predicted_sales": predicted_sales
    })

    # Use the forecast as input for the next forecast month.
    sales_history.append(predicted_sales)

future_forecast = pd.DataFrame(future_results)

future_forecast.to_csv(
    OUTPUT_PATH / "next_3_months_sales_forecast.csv",
    index=False
)

print("\n========== NEXT 3 MONTHS SALES FORECAST ==========")
print(future_forecast.to_string(index=False))

# -------------------------------------------------
# 11. SAVE TRAINED MODEL
# -------------------------------------------------

joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns
    },
    MODEL_PATH / "sales_forecasting_model.pkl"
)

# -------------------------------------------------
# 12. ACTUAL VS PREDICTED CHART
# -------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    test_results["month"],
    test_results["actual_sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    test_results["month"],
    test_results["predicted_sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Product Sales (BRL)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(CHART_PATH / "actual_vs_predicted_sales.png")
plt.close()

# -------------------------------------------------
# 13. HISTORICAL SALES AND FUTURE FORECAST CHART
# -------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["month"],
    monthly_sales["sales"],
    marker="o",
    label="Historical Sales"
)

future_x = (
    [monthly_sales["month"].iloc[-1]]
    + future_forecast["month"].tolist()
)

future_y = (
    [monthly_sales["sales"].iloc[-1]]
    + future_forecast["predicted_sales"].tolist()
)

plt.plot(
    future_x,
    future_y,
    marker="o",
    linestyle="--",
    label="Future Forecast"
)

plt.title("Historical Sales and Next 3 Months Forecast")
plt.xlabel("Month")
plt.ylabel("Product Sales (BRL)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(CHART_PATH / "future_sales_forecast.png")
plt.close()

# -------------------------------------------------
# 14. MODEL COMPARISON CHART
# -------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["RMSE"]
)

plt.title("Sales Forecasting Model Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE (Lower is Better)")
plt.tight_layout()

plt.savefig(CHART_PATH / "forecast_model_comparison.png")
plt.close()

# -------------------------------------------------
# 15. COMPLETION MESSAGE
# -------------------------------------------------

print("\n========== SALES FORECASTING COMPLETED ==========")
print("Historical monthly sales generated.")
print("Forecasting models trained and evaluated.")
print("Next 3 months sales forecast generated.")
print("Forecasting model saved successfully.")

print("\nReports saved in:", OUTPUT_PATH)
print("Charts saved in:", CHART_PATH)
print("Model saved in:", MODEL_PATH)
