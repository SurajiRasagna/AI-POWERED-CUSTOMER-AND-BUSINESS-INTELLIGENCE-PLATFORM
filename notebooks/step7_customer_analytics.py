
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# 1. PROJECT PATHS
# -------------------------------------------------

DATA_PATH = Path("data/processed")
OUTPUT_PATH = DATA_PATH / "customer_analytics_outputs"
CHART_PATH = OUTPUT_PATH / "charts"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CHART_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 2. LOAD CLEANED DATASETS
# -------------------------------------------------

orders = pd.read_csv(DATA_PATH / "orders_cleaned.csv")
order_items = pd.read_csv(DATA_PATH / "order_items_cleaned.csv")
customers = pd.read_csv(DATA_PATH / "customers_cleaned.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"], errors="coerce"
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

# Connect orders to customer information
customer_orders = delivered_orders.merge(
    customers[["customer_id", "customer_unique_id", "customer_state"]],
    on="customer_id",
    how="left"
)

# Connect orders to product prices
customer_items = customer_orders.merge(
    order_items[["order_id", "price"]],
    on="order_id",
    how="inner"
)

# -------------------------------------------------
# 4. BASIC CUSTOMER ANALYSIS
# -------------------------------------------------

unique_customers = customer_orders["customer_unique_id"].nunique()
total_delivered_orders = delivered_orders["order_id"].nunique()

customer_order_counts = (
    customer_orders.groupby("customer_unique_id")["order_id"]
    .nunique()
)

repeat_customers = (customer_order_counts > 1).sum()
one_time_customers = (customer_order_counts == 1).sum()

basic_summary = pd.DataFrame({
    "Metric": [
        "Unique Customers",
        "Total Delivered Orders",
        "Repeat Customers",
        "One-Time Customers"
    ],
    "Value": [
        unique_customers,
        total_delivered_orders,
        repeat_customers,
        one_time_customers
    ]
})

basic_summary.to_csv(
    OUTPUT_PATH / "customer_summary.csv", index=False
)

print("\n========== CUSTOMER SUMMARY ==========")
print(basic_summary.to_string(index=False))

# -------------------------------------------------
# 5. CALCULATE RFM VALUES
# -------------------------------------------------

# Use the day after the latest recorded purchase as
# the reference date for recency.
reference_date = (
    delivered_orders["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)

# Recency: days since the customer's last delivered order
last_purchase = (
    customer_orders.groupby("customer_unique_id")
    ["order_purchase_timestamp"].max()
)

recency = (reference_date - last_purchase).dt.days

# Frequency: number of distinct delivered orders
frequency = (
    customer_orders.groupby("customer_unique_id")["order_id"]
    .nunique()
)

# Monetary: total product price across delivered orders
monetary = (
    customer_items.groupby("customer_unique_id")["price"]
    .sum()
)

rfm = pd.concat(
    [recency, frequency, monetary], axis=1
).reset_index()

rfm.columns = [
    "customer_unique_id",
    "Recency",
    "Frequency",
    "Monetary"
]

rfm = rfm.dropna(
    subset=["Recency", "Frequency", "Monetary"]
)

# -------------------------------------------------
# 6. GENERATE RFM SCORES
# -------------------------------------------------

# Ranking first helps avoid duplicate quantile boundaries.
# A score of 5 is best for all three dimensions.
rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

# -------------------------------------------------
# 7. CUSTOMER SEGMENTATION
# -------------------------------------------------

def assign_segment(row):
    if row["R_Score"] >= 4 and row["F_Score"] >= 4:
        return "Champions"
    elif row["F_Score"] >= 4:
        return "Loyal Customers"
    elif row["R_Score"] >= 4:
        return "Recent Customers"
    elif row["R_Score"] <= 2 and row["F_Score"] <= 2:
        return "At Risk / Inactive"
    else:
        return "Regular Customers"

rfm["Customer_Segment"] = rfm.apply(
    assign_segment, axis=1
)

rfm.to_csv(
    OUTPUT_PATH / "customer_rfm_segmentation.csv",
    index=False
)

print("\n========== RFM CUSTOMER ANALYSIS ==========")
print(rfm.head(10).to_string(index=False))

# -------------------------------------------------
# 8. CUSTOMER SEGMENT SUMMARY
# -------------------------------------------------

segment_summary = (
    rfm.groupby("Customer_Segment")
    .agg(
        Customer_Count=("customer_unique_id", "count"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Total_Monetary=("Monetary", "sum")
    )
    .reset_index()
)

segment_summary.to_csv(
    OUTPUT_PATH / "customer_segment_summary.csv",
    index=False
)

print("\n========== CUSTOMER SEGMENT SUMMARY ==========")
print(segment_summary.to_string(index=False))

# -------------------------------------------------
# 9. TOP 10 HIGH-VALUE CUSTOMERS
# -------------------------------------------------

top_customers = (
    rfm.sort_values("Monetary", ascending=False)
    .head(10)
)

top_customers.to_csv(
    OUTPUT_PATH / "top_10_customers.csv",
    index=False
)

# -------------------------------------------------
# 10. CUSTOMER STATE ANALYSIS
# -------------------------------------------------

# One row per unique customer to avoid counting
# customers more than once.
unique_customer_details = (
    customer_orders[
        ["customer_unique_id", "customer_state"]
    ]
    .dropna(subset=["customer_unique_id"])
    .drop_duplicates(subset=["customer_unique_id"])
)

state_summary = (
    unique_customer_details.groupby("customer_state")
    ["customer_unique_id"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="customer_count")
)

state_summary.to_csv(
    OUTPUT_PATH / "customer_state_analysis.csv",
    index=False
)

# -------------------------------------------------
# 11. GENERATE CHARTS
# -------------------------------------------------

# Chart 1: Customer segment distribution
segment_counts = rfm["Customer_Segment"].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(segment_counts.index, segment_counts.values)
plt.title("Customer Segment Distribution")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(CHART_PATH / "customer_segments.png")
plt.close()

# Chart 2: Top 10 high-value customers
plt.figure(figsize=(12, 6))
plt.bar(
    top_customers["customer_unique_id"].astype(str),
    top_customers["Monetary"]
)
plt.title("Top 10 High-Value Customers")
plt.xlabel("Customer ID")
plt.ylabel("Total Product Spending (BRL)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(CHART_PATH / "top_10_customers.png")
plt.close()

# Chart 3: Customer distribution by state
plt.figure(figsize=(12, 6))
plt.bar(
    state_summary["customer_state"].astype(str),
    state_summary["customer_count"]
)
plt.title("Customer Distribution by State")
plt.xlabel("Customer State")
plt.ylabel("Number of Unique Customers")
plt.tight_layout()
plt.savefig(CHART_PATH / "customers_by_state.png")
plt.close()

# Chart 4: One-time vs repeat customers
plt.figure(figsize=(8, 6))
plt.bar(
    ["One-Time Customers", "Repeat Customers"],
    [one_time_customers, repeat_customers]
)
plt.title("One-Time vs Repeat Customers")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(CHART_PATH / "repeat_vs_one_time_customers.png")
plt.close()

# -------------------------------------------------
# 12. COMPLETION MESSAGE
# -------------------------------------------------

print("\n========== CUSTOMER ANALYTICS COMPLETED ==========")
print("Basic customer analysis generated.")
print("RFM analysis completed.")
print("Customer segmentation completed.")
print("Top 10 customer analysis generated.")
print("Customer state analysis generated.")
print("\nReports saved in:", OUTPUT_PATH)
print("Charts saved in:", CHART_PATH)
