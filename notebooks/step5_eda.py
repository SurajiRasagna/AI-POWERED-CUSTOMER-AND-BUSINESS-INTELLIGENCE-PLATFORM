
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Define folder paths
DATA_PATH = Path("data/processed")
OUTPUT_PATH = Path("data/processed/eda_outputs")
CHART_PATH = OUTPUT_PATH / "charts"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CHART_PATH.mkdir(parents=True, exist_ok=True)

# Step 2: Load cleaned datasets
customers = pd.read_csv(DATA_PATH / "customers_cleaned.csv")
orders = pd.read_csv(DATA_PATH / "orders_cleaned.csv")
order_items = pd.read_csv(DATA_PATH / "order_items_cleaned.csv")
reviews = pd.read_csv(DATA_PATH / "reviews_cleaned.csv")
products = pd.read_csv(DATA_PATH / "products_cleaned.csv")
payments = pd.read_csv(DATA_PATH / "payments_cleaned.csv")
category_translation = pd.read_csv(
    DATA_PATH / "category_translation_cleaned.csv"
)

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# Step 3: General dataset information
print("\nCUSTOMER INFORMATION")
print("Total customer records:", len(customers))
print("Unique customers:", customers["customer_unique_id"].nunique())

print("\nORDER INFORMATION")
print("Total orders:", orders["order_id"].nunique())
print("\nOrder status:")
print(orders["order_status"].value_counts())

print("\nPRODUCT INFORMATION")
print("Total products:", products["product_id"].nunique())

# Step 4: Sales analysis
# Calculate product sales from order item prices
total_sales = order_items["price"].sum()
total_freight = order_items["freight_value"].sum()

print("\n========== SALES ANALYSIS ==========")
print("Total product sales (BRL):", round(total_sales, 2))
print("Total freight value (BRL):", round(total_freight, 2))

# Save sales summary
sales_summary = pd.DataFrame({
    "Metric": ["Total Product Sales", "Total Freight Value"],
    "Value_BRL": [total_sales, total_freight]
})

sales_summary.to_csv(
    OUTPUT_PATH / "sales_summary.csv",
    index=False
)

# Step 5: Monthly sales analysis
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"],
    errors="coerce"
)

sales_data = order_items.merge(
    orders[["order_id", "order_purchase_timestamp", "order_status"]],
    on="order_id",
    how="inner"
)

# Use delivered orders for completed-order sales trends
delivered_sales = sales_data[
    sales_data["order_status"] == "delivered"
].copy()

delivered_sales["month"] = (
    delivered_sales["order_purchase_timestamp"].dt.to_period("M").astype(str)
)

monthly_sales = delivered_sales.groupby("month")["price"].sum().reset_index()

monthly_sales.rename(
    columns={"price": "product_sales"},
    inplace=True
)

monthly_sales.to_csv(
    OUTPUT_PATH / "monthly_sales.csv",
    index=False
)

print("\nMONTHLY SALES")
print(monthly_sales.head(10))

# Generate monthly sales chart
plt.figure(figsize=(12, 6))
plt.plot(
    monthly_sales["month"],
    monthly_sales["product_sales"],
    marker="o"
)
plt.title("Monthly Product Sales Trend")
plt.xlabel("Month")
plt.ylabel("Product Sales (BRL)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(CHART_PATH / "monthly_sales.png")
plt.close()

# Step 6: Top-selling product categories
products_with_category = products.merge(
    category_translation,
    on="product_category_name",
    how="left"
)

products_with_category["product_category_name_english"] = (
    products_with_category["product_category_name_english"]
    .fillna("Unknown")
)

category_sales = order_items.merge(
    products_with_category[
        ["product_id", "product_category_name_english"]
    ],
    on="product_id",
    how="left"
)

top_categories = (
    category_sales.groupby("product_category_name_english")["price"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

top_categories.rename(
    columns={"price": "total_sales"},
    inplace=True
)

top_categories.to_csv(
    OUTPUT_PATH / "top_categories.csv",
    index=False
)

print("\nTOP 10 PRODUCT CATEGORIES")
print(top_categories)

# Generate top categories chart
plt.figure(figsize=(12, 6))
plt.barh(
    top_categories["product_category_name_english"],
    top_categories["total_sales"]
)
plt.title("Top 10 Product Categories by Sales")
plt.xlabel("Sales (BRL)")
plt.ylabel("Product Category")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(CHART_PATH / "top_categories.png")
plt.close()

# Step 7: Order status analysis
order_status = orders["order_status"].value_counts().reset_index()
order_status.columns = ["order_status", "order_count"]

order_status.to_csv(
    OUTPUT_PATH / "order_status_summary.csv",
    index=False
)

print("\nORDER STATUS SUMMARY")
print(order_status)

# Generate order status chart
plt.figure(figsize=(10, 5))
plt.bar(
    order_status["order_status"],
    order_status["order_count"]
)
plt.title("Order Status Distribution")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(CHART_PATH / "order_status.png")
plt.close()

# Step 8: Customer review analysis
review_summary = (
    reviews["review_score"]
    .value_counts()
    .sort_index()
    .reset_index()
)

review_summary.columns = ["review_score", "review_count"]

review_summary.to_csv(
    OUTPUT_PATH / "review_summary.csv",
    index=False
)

print("\nCUSTOMER REVIEW SUMMARY")
print(review_summary)

print(
    "\nAverage review score:",
    round(reviews["review_score"].mean(), 2)
)

# Generate review score chart
plt.figure(figsize=(8, 5))
plt.bar(
    review_summary["review_score"],
    review_summary["review_count"]
)
plt.title("Customer Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig(CHART_PATH / "review_scores.png")
plt.close()

# Step 9: Final result
print("\n========== EDA COMPLETED ==========")
print("Sales analysis completed.")
print("Monthly sales analysis completed.")
print("Product category analysis completed.")
print("Order status analysis completed.")
print("Customer review analysis completed.")

print("\nReports saved in:", OUTPUT_PATH)
print("Charts saved in:", CHART_PATH)
