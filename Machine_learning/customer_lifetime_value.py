import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

BASE_DIR = r"C:\Users\suraj\OneDrive\AI_Customer_BI_Project"

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

OUTPUT_DIR = os.path.join(
    DATA_DIR, "customer_lifetime_value_outputs"
)

CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ==========================================
# STEP 2: LOAD DATASETS
# ==========================================

customers_path = os.path.join(DATA_DIR, "customers_cleaned.csv")
orders_path = os.path.join(DATA_DIR, "orders_cleaned.csv")
payments_path = os.path.join(DATA_DIR, "payments_cleaned.csv")

for file_path in [customers_path, orders_path, payments_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )

customers = pd.read_csv(customers_path)
orders = pd.read_csv(orders_path)
payments = pd.read_csv(payments_path)

print("Datasets loaded successfully!")
print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Payments:", payments.shape)

# ==========================================
# STEP 3: CHECK REQUIRED COLUMNS
# ==========================================

required_columns = {
    "customers": ["customer_id"],
    "orders": [
        "order_id",
        "customer_id",
        "order_purchase_timestamp"
    ],
    "payments": ["order_id", "payment_value"]
}

for name, columns in required_columns.items():
    dataframe = {
        "customers": customers,
        "orders": orders,
        "payments": payments
    }[name]

    missing = [col for col in columns if col not in dataframe.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {name}: {missing}"
        )

# ==========================================
# STEP 4: CLEAN DATA
# ==========================================

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

payments["payment_value"] = pd.to_numeric(
    payments["payment_value"],
    errors="coerce"
)

orders = orders.dropna(
    subset=["order_id", "customer_id", "order_purchase_timestamp"]
)

payments = payments.dropna(
    subset=["order_id", "payment_value"]
)

# Exclude canceled and unavailable orders when status is available
if "order_status" in orders.columns:
    orders = orders[
        ~orders["order_status"].astype(str).str.lower().isin(
            ["canceled", "unavailable"]
        )
    ]

# ==========================================
# STEP 5: CALCULATE ORDER PAYMENT
# ==========================================

order_payments = (
    payments.groupby("order_id", as_index=False)
    ["payment_value"].sum()
)

# ==========================================
# STEP 6: MERGE CUSTOMER, ORDER AND PAYMENT DATA
# ==========================================

customer_orders = orders.merge(
    customers,
    on="customer_id",
    how="left"
)

customer_orders = customer_orders.merge(
    order_payments,
    on="order_id",
    how="inner"
)

# Use unique customer ID when available
if (
    "customer_unique_id" in customer_orders.columns
    and customer_orders["customer_unique_id"].notna().any()
):
    customer_key = "customer_unique_id"
else:
    customer_key = "customer_id"

customer_orders = customer_orders.dropna(
    subset=[customer_key, "payment_value"]
)

print("\nCustomer, order and payment data merged successfully!")

# ==========================================
# STEP 7: CALCULATE CUSTOMER METRICS
# ==========================================

reference_date = (
    customer_orders["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)

customer_metrics = customer_orders.groupby(
    customer_key
).agg(
    total_spending=("payment_value", "sum"),
    total_orders=("order_id", "nunique"),
    last_purchase=("order_purchase_timestamp", "max"),
    first_purchase=("order_purchase_timestamp", "min")
).reset_index()

customer_metrics["average_order_value"] = (
    customer_metrics["total_spending"]
    / customer_metrics["total_orders"]
)

customer_metrics["recency_days"] = (
    reference_date - customer_metrics["last_purchase"]
).dt.days

customer_metrics["customer_lifespan_days"] = (
    customer_metrics["last_purchase"]
    - customer_metrics["first_purchase"]
).dt.days

# Historical CLV proxy based on observed spending
customer_metrics["historical_clv"] = (
    customer_metrics["total_spending"]
)

# ==========================================
# STEP 8: CUSTOMER VALUE SEGMENTATION
# ==========================================

customer_metrics["customer_segment"] = "Regular"

if len(customer_metrics) >= 3:
    ranked_spending = customer_metrics[
        "total_spending"
    ].rank(method="first")

    customer_metrics["customer_segment"] = pd.qcut(
        ranked_spending,
        q=3,
        labels=[
            "Low Value",
            "Medium Value",
            "High Value"
        ]
    )

# ==========================================
# STEP 9: SAVE CUSTOMER CLV RESULTS
# ==========================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "customer_lifetime_value_analysis.csv"
)

customer_metrics.to_csv(output_csv, index=False)

print("\nCustomer Lifetime Value Analysis:")
print(customer_metrics.head(10).to_string(index=False))

# ==========================================
# STEP 10: CUSTOMER SEGMENT SUMMARY
# ==========================================

segment_summary = customer_metrics.groupby(
    "customer_segment",
    observed=False
).agg(
    customer_count=(customer_key, "count"),
    total_revenue=("total_spending", "sum"),
    average_customer_value=("historical_clv", "mean")
).reset_index()

segment_csv = os.path.join(
    OUTPUT_DIR,
    "customer_value_segment_summary.csv"
)

segment_summary.to_csv(segment_csv, index=False)

print("\nCustomer Segment Summary:")
print(segment_summary.to_string(index=False))

# ==========================================
# STEP 11: TOP 10 HIGH-VALUE CUSTOMERS
# ==========================================

top_customers = customer_metrics.nlargest(
    10, "historical_clv"
)

top_csv = os.path.join(
    OUTPUT_DIR,
    "top_10_high_value_customers.csv"
)

top_customers.to_csv(top_csv, index=False)

# ==========================================
# STEP 12: GENERATE CHARTS
# ==========================================

sns.set_theme(style="whitegrid")

# Chart 1: Top 10 customers by historical CLV
plt.figure(figsize=(12, 6))

sns.barplot(
    data=top_customers,
    x="historical_clv",
    y=customer_key
)

plt.title("Top 10 Customers by Historical CLV")
plt.xlabel("Historical Customer Value")
plt.ylabel("Customer ID")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "top_10_customer_clv.png")
)

plt.close()

# Chart 2: Customer segment distribution
plt.figure(figsize=(9, 6))

sns.countplot(
    data=customer_metrics,
    x="customer_segment",
    order=["Low Value", "Medium Value", "High Value"]
)

plt.title("Customer Value Segment Distribution")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "customer_segment_distribution.png")
)

plt.close()

# Chart 3: Spending distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    data=customer_metrics,
    x="total_spending",
    bins=30,
    kde=True
)

plt.title("Customer Spending Distribution")
plt.xlabel("Total Customer Spending")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_DIR, "customer_spending_distribution.png")
)

plt.close()

# ==========================================
# STEP 13: FINAL OUTPUT
# ==========================================

print("\n======================================")
print("CUSTOMER LIFETIME VALUE ANALYSIS DONE")
print("======================================")

print("\nTotal Customers Analyzed:", len(customer_metrics))

print("\nOutput Folder:")
print(OUTPUT_DIR)

print("\nCharts Folder:")
print(CHART_DIR)

print("\nGenerated Files:")
print("1. customer_lifetime_value_analysis.csv")
print("2. customer_value_segment_summary.csv")
print("3. top_10_high_value_customers.csv")
print("4. top_10_customer_clv.png")
print("5. customer_segment_distribution.png")
print("6. customer_spending_distribution.png")