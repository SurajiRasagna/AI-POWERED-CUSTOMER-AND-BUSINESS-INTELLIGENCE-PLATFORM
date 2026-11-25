
import pandas as pd
from pathlib import Path

# Step 1: Set the dataset folder path
DATA_PATH = Path("data/raw/raw")

# Step 2: Load all nine CSV datasets
customers = pd.read_csv(DATA_PATH / "olist_customers_dataset.csv")

orders = pd.read_csv(DATA_PATH / "olist_orders_dataset.csv")

order_items = pd.read_csv(DATA_PATH / "olist_order_items_dataset.csv")

payments = pd.read_csv(DATA_PATH / "olist_order_payments_dataset.csv")

reviews = pd.read_csv(DATA_PATH / "olist_order_reviews_dataset.csv")

products = pd.read_csv(DATA_PATH / "olist_products_dataset.csv")

sellers = pd.read_csv(DATA_PATH / "olist_sellers_dataset.csv")

geolocation = pd.read_csv(DATA_PATH / "olist_geolocation_dataset.csv")

category_translation = pd.read_csv(
    DATA_PATH / "product_category_name_translation.csv"
)

# Step 3: Store all datasets in one dictionary
datasets = {
    "Customers": customers,
    "Orders": orders,
    "Order Items": order_items,
    "Payments": payments,
    "Reviews": reviews,
    "Products": products,
    "Sellers": sellers,
    "Geolocation": geolocation,
    "Category Translation": category_translation
}

# Step 4: Display dataset information
print("\n========== DATASET LOADING SUMMARY ==========")

for name, df in datasets.items():
    print("\nDataset:", name)
    print("Number of Rows:", df.shape[0])
    print("Number of Columns:", df.shape[1])

print("\nAll nine datasets loaded successfully!")

# Step 5: Display first five customer records
print("\n========== CUSTOMER DATA ==========")
print(customers.head())

# Step 6: Display first five order records
print("\n========== ORDER DATA ==========")
print(orders.head())

# Step 7: Display first five product records
print("\n========== PRODUCT DATA ==========")
print(products.head())

# Step 8: Check missing values
print("\n========== MISSING VALUES ==========")

for name, df in datasets.items():
    print("\n", name)
    print(df.isnull().sum())
