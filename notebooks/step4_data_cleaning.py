
import pandas as pd
from pathlib import Path

# Step 1: Define project folders
RAW_PATH = Path("data/raw/raw")
PROCESSED_PATH = Path("data/processed")

# Create processed folder if it does not exist
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# Step 2: Load all datasets
customers = pd.read_csv(RAW_PATH / "olist_customers_dataset.csv")
orders = pd.read_csv(RAW_PATH / "olist_orders_dataset.csv")
order_items = pd.read_csv(RAW_PATH / "olist_order_items_dataset.csv")
payments = pd.read_csv(RAW_PATH / "olist_order_payments_dataset.csv")
reviews = pd.read_csv(RAW_PATH / "olist_order_reviews_dataset.csv")
products = pd.read_csv(RAW_PATH / "olist_products_dataset.csv")
sellers = pd.read_csv(RAW_PATH / "olist_sellers_dataset.csv")
geolocation = pd.read_csv(RAW_PATH / "olist_geolocation_dataset.csv")
category_translation = pd.read_csv(
    RAW_PATH / "product_category_name_translation.csv"
)

# Step 3: Store datasets in a dictionary
datasets = {
    "customers": customers,
    "orders": orders,
    "order_items": order_items,
    "payments": payments,
    "reviews": reviews,
    "products": products,
    "sellers": sellers,
    "geolocation": geolocation,
    "category_translation": category_translation
}

print("========== DATA CLEANING STARTED ==========")

# Step 4: Remove exact duplicate records
for name, df in datasets.items():
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)

    print(f"{name}: Removed {before - after} duplicate rows")

# Step 5: Clean customer data
for col in [
    "customer_zip_code_prefix",
    "customer_city",
    "customer_state"
]:
    customers[col] = customers[col].fillna("Unknown")

# Step 6: Clean order data
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")

# Step 7: Clean order item data
order_items["price"] = pd.to_numeric(
    order_items["price"], errors="coerce"
)
order_items["freight_value"] = pd.to_numeric(
    order_items["freight_value"], errors="coerce"
)

# Price and freight are important for sales calculations
order_items.dropna(
    subset=["order_id", "product_id", "price", "freight_value"],
    inplace=True
)

# Step 8: Clean payment data
payments["payment_value"] = pd.to_numeric(
    payments["payment_value"], errors="coerce"
)

payments.dropna(subset=["payment_value"], inplace=True)

payments["payment_type"] = payments["payment_type"].fillna("Unknown")

payments["payment_installments"] = pd.to_numeric(
    payments["payment_installments"], errors="coerce"
)

# Step 9: Clean review data
reviews["review_score"] = pd.to_numeric(
    reviews["review_score"], errors="coerce"
)

reviews["review_comment_message"] = (
    reviews["review_comment_message"].fillna("")
)

# Step 10: Clean product data
products["product_category_name"] = (
    products["product_category_name"].fillna("Unknown")
)

product_numeric_columns = [
    "product_name_lenght",
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm"
]

for col in product_numeric_columns:
    products[col] = pd.to_numeric(products[col], errors="coerce")

    if products[col].notna().any():
        products[col] = products[col].fillna(products[col].median())

# Step 11: Clean seller data
for col in [
    "seller_zip_code_prefix",
    "seller_city",
    "seller_state"
]:
    sellers[col] = sellers[col].fillna("Unknown")

# Step 12: Clean geolocation data
geolocation["geolocation_lat"] = pd.to_numeric(
    geolocation["geolocation_lat"], errors="coerce"
)
geolocation["geolocation_lng"] = pd.to_numeric(
    geolocation["geolocation_lng"], errors="coerce"
)

# Remove records without valid coordinates
geolocation.dropna(
    subset=["geolocation_lat", "geolocation_lng"],
    inplace=True
)

geolocation["geolocation_city"] = (
    geolocation["geolocation_city"].fillna("Unknown")
)
geolocation["geolocation_state"] = (
    geolocation["geolocation_state"].fillna("Unknown")
)

# Step 13: Clean category translation data
category_translation.dropna(
    subset=["product_category_name", "product_category_name_english"],
    inplace=True
)

# Step 14: Save all cleaned datasets
print("\n========== SAVING CLEANED DATASETS ==========")

for name, df in datasets.items():
    output_file = PROCESSED_PATH / f"{name}_cleaned.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file} | Rows: {len(df)}")

# Step 15: Final summary
print("\n========== DATA CLEANING COMPLETED ==========")

for name, df in datasets.items():
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nAll cleaned datasets saved successfully!")

