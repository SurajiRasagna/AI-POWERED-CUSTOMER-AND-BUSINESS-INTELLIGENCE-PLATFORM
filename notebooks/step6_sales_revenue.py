
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# 1. SET PROJECT PATHS
# -------------------------------------------------

DATA_PATH = Path("data/processed")
OUTPUT_PATH = DATA_PATH / "sales_revenue_outputs"
CHART_PATH = OUTPUT_PATH / "charts"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CHART_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 2. LOAD CLEANED DATASETS
# -------------------------------------------------

orders = pd.read_csv(DATA_PATH / "orders_cleaned.csv")
order_items = pd.read_csv(DATA_PATH / "order_items_cleaned.csv")
products = pd.read_csv(DATA_PATH / "products_cleaned.csv")
customers = pd.read_csv(DATA_PATH / "customers_cleaned.csv")
payments = pd.read_csv(DATA_PATH / "payments_cleaned.csv")
category_translation = pd.read_csv(
    DATA_PATH / "category_translation_cleaned.csv"
)

# Convert date column
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"], errors="coerce"
)

# Ensure numeric columns
for col in ["price", "freight_value"]:
    order_items[col] = pd.to_numeric(
        order_items[col], errors="coerce"
    ).fillna(0)

payments["payment_value"] = pd.to_numeric(
    payments["payment_value"], errors="coerce"
).fillna(0)

# -------------------------------------------------
# 3. FILTER DELIVERED ORDERS
# -------------------------------------------------

delivered_orders = orders[
    orders["order_status"] == "delivered"
].copy()

delivered_items = order_items.merge(
    delivered_orders[
        ["order_id", "customer_id", "order_purchase_timestamp"]
    ],
    on="order_id",
    how="inner"
)

# Calculate item-level sales
delivered_items["total_item_value"] = (
    delivered_items["price"] + delivered_items["freight_value"]
)

# -------------------------------------------------
# 4. OVERALL SALES SUMMARY
# -------------------------------------------------

total_product_sales = delivered_items["price"].sum()
total_freight = delivered_items["freight_value"].sum()
total_sales_including_freight = (
    total_product_sales + total_freight
)

order_values = delivered_items.groupby("order_id")[
    "total_item_value"
].sum()

average_order_value = order_values.mean() if len(order_values) else 0
total_delivered_orders = delivered_orders["order_id"].nunique()

sales_summary = pd.DataFrame({
    "Metric": [
        "Total Delivered Orders",
        "Total Product Sales",
        "Total Freight Value",
        "Total Sales Including Freight",
        "Average Order Value"
    ],
    "Value": [
        total_delivered_orders,
        total_product_sales,
        total_freight,
        total_sales_including_freight,
        average_order_value
    ]
})

sales_summary.to_csv(
    OUTPUT_PATH / "overall_sales_summary.csv", index=False
)

print("\n========== OVERALL SALES SUMMARY ==========")
print(sales_summary.to_string(index=False))

# -------------------------------------------------
# 5. MONTHLY SALES ANALYSIS
# -------------------------------------------------

delivered_items["month"] = (
    delivered_items["order_purchase_timestamp"].dt.to_period("M").astype(str)
)

monthly_sales = (
    delivered_items.groupby("month")
    .agg(
        product_sales=("price", "sum"),
        freight_value=("freight_value", "sum"),
        total_sales=("total_item_value", "sum")
    )
    .reset_index()
)

monthly_sales.to_csv(
    OUTPUT_PATH / "monthly_sales_revenue.csv", index=False
)

# Monthly sales chart
plt.figure(figsize=(12, 6))
plt.plot(
    monthly_sales["month"],
    monthly_sales["total_sales"],
    marker="o"
)
plt.title("Monthly Sales Revenue")
plt.xlabel("Month")
plt.ylabel("Sales Value (BRL)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(CHART_PATH / "monthly_sales_revenue.png")
plt.close()

# -------------------------------------------------
# 6. TOP 10 PRODUCT ANALYSIS
# -------------------------------------------------

top_products = (
    delivered_items.groupby("product_id")["price"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="product_sales")
)

top_products = top_products.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)

top_products.to_csv(
    OUTPUT_PATH / "top_10_products.csv", index=False
)

plt.figure(figsize=(12, 6))
plt.bar(
    top_products["product_id"].astype(str),
    top_products["product_sales"]
)
plt.title("Top 10 Products by Sales")
plt.xlabel("Product ID")
plt.ylabel("Product Sales (BRL)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(CHART_PATH / "top_10_products.png")
plt.close()

# -------------------------------------------------
# 7. PRODUCT CATEGORY ANALYSIS
# -------------------------------------------------

category_data = delivered_items.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)

category_data = category_data.merge(
    category_translation,
    on="product_category_name",
    how="left"
)

category_column = (
    "product_category_name_english"
    if "product_category_name_english" in category_data.columns
    else "product_category_name"
)

category_data[category_column] = (
    category_data[category_column].fillna("Unknown")
)

top_categories = (
    category_data.groupby(category_column)["price"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name="product_sales")
)

top_categories.to_csv(
    OUTPUT_PATH / "top_10_categories.csv", index=False
)

plt.figure(figsize=(12, 6))
plt.barh(
    top_categories[category_column],
    top_categories["product_sales"]
)
plt.title("Top 10 Product Categories by Sales")
plt.xlabel("Product Sales (BRL)")
plt.ylabel("Product Category")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(CHART_PATH / "top_10_categories.png")
plt.close()

# -------------------------------------------------
# 8. STATE-WISE SALES ANALYSIS
# -------------------------------------------------

state_data = delivered_items.merge(
    customers[["customer_id", "customer_state"]],
    on="customer_id",
    how="left"
)

state_sales = (
    state_data.groupby("customer_state")["price"]
    .sum()
    .sort_values(ascending=False)
    .reset_index(name="product_sales")
)

state_sales.to_csv(
    OUTPUT_PATH / "state_wise_sales.csv", index=False
)

plt.figure(figsize=(12, 6))
plt.bar(
    state_sales["customer_state"].astype(str),
    state_sales["product_sales"]
)
plt.title("State-wise Product Sales")
plt.xlabel("Customer State")
plt.ylabel("Product Sales (BRL)")
plt.tight_layout()
plt.savefig(CHART_PATH / "state_wise_sales.png")
plt.close()

# -------------------------------------------------
# 9. PAYMENT METHOD ANALYSIS
# -------------------------------------------------

delivered_payment_ids = delivered_orders[["order_id"]].drop_duplicates()

delivered_payments = payments.merge(
    delivered_payment_ids,
    on="order_id",
    how="inner"
)

payment_analysis = (
    delivered_payments.groupby("payment_type")["payment_value"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

payment_analysis.to_csv(
    OUTPUT_PATH / "payment_method_analysis.csv", index=False
)

plt.figure(figsize=(9, 6))
plt.bar(
    payment_analysis["payment_type"].astype(str),
    payment_analysis["payment_value"]
)
plt.title("Payment Method Analysis")
plt.xlabel("Payment Method")
plt.ylabel("Payment Value (BRL)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(CHART_PATH / "payment_method_analysis.png")
plt.close()

# -------------------------------------------------
# 10. COMPLETION MESSAGE
# -------------------------------------------------

print("\n========== SALES AND REVENUE ANALYTICS COMPLETED ==========")
print("Overall sales summary generated.")
print("Monthly sales analysis generated.")
print("Top 10 products analysis generated.")
print("Top 10 product categories analysis generated.")
print("State-wise sales analysis generated.")
print("Payment method analysis generated.")

print("\nReports saved in:", OUTPUT_PATH)
print("Charts saved in:", CHART_PATH)
