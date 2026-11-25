# Live Demand Prediction

This page turns the batch forecasting project into an operational tool. It
loads the trained per-product RandomForest models from
`models/product_demand_forecasting_models.pkl` and scores new demand figures
instantly — no retraining required.

## Run the dashboard

```powershell
cd C:\Users\suraj\OneDrive\AI_Customer_BI_Project
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Then open **🔮 Live Demand Prediction** in the sidebar.

## What the page does

1. **Product selector** — only products that have a trained model are listed,
   labelled with their category and latest monthly units.
2. **Current demand inputs** — `lag_1` (latest month), `lag_2` and `lag_3`
   (the two prior months). These are pre-filled from the dataset and can be
   overwritten with the newest real figures.
3. **Forecast month** — choose the month to predict (next 1–6 months).
4. **Predict demand** — returns:
   - predicted units for the selected month,
   - a demand signal (High / Normal / Low vs the recent average),
   - safety stock at roughly a 95% service level
     (`1.65 × std of test-period residuals`),
   - reorder point (`forecast + safety stock`).
5. **Next months** — a recursive 1–3 month forecast, a chart of the last 12
   actual months plus the forecast, and a comparison with the Step 10 batch
   forecast file.
6. **Score a live demand file** — upload a CSV with:

   | product_id | month      | quantity_sold |
   |------------|------------|---------------|
   | abc123...  | 2018-08-01 | 42            |

   The latest three rows per product are used to predict the following
   month. Rows are skipped with an explicit status when a product has no
   model or fewer than three months of history. Results can be downloaded
   as CSV.

## Model contract

The page reproduces the exact feature engineering used in
`notebooks/step10_product_demand_forecasting.py`:

```
features: time_index, month_number, lag_1, lag_2, lag_3, rolling_mean_3
target:   quantity_sold
```

`time_index` defaults to the number of months in the product's history and
can be overridden under **Advanced model inputs**. Predictions are clipped at
zero so negative demand is never returned.

## Updating the models

When new sales data arrives:

1. Append the new orders to `data/processed/orders_cleaned.csv` and
   `order_items_cleaned.csv`.
2. Re-run `notebooks/step10_product_demand_forecasting.py`.
3. Restart Streamlit (or clear the cache) so the updated `.pkl` is loaded.

## Files

| File | Purpose |
|------|---------|
| `dashboard/app.py` | Dashboard UI and the live prediction page |
| `dashboard/live_prediction.py` | Prediction, batch scoring and planning logic |
| `models/product_demand_forecasting_models.pkl` | Trained per-product models |
| `data/processed/product_demand_forecasting_outputs/monthly_product_demand.csv` | Default history used by the page |
| `data/processed/product_demand_forecasting_outputs/product_demand_test_predictions.csv` | Residuals used for safety stock |
