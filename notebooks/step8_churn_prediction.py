
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -------------------------------------------------
# 1. PROJECT PATHS
# -------------------------------------------------

DATA_PATH = Path("data/processed")
OUTPUT_PATH = DATA_PATH / "churn_prediction_outputs"
CHART_PATH = OUTPUT_PATH / "charts"
MODEL_PATH = Path("models")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
CHART_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# 2. LOAD CLEANED DATA
# -------------------------------------------------

orders = pd.read_csv(DATA_PATH / "orders_cleaned.csv")
customers = pd.read_csv(DATA_PATH / "customers_cleaned.csv")
order_items = pd.read_csv(DATA_PATH / "order_items_cleaned.csv")

orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"], errors="coerce"
)

order_items["price"] = pd.to_numeric(
    order_items["price"], errors="coerce"
).fillna(0)

# -------------------------------------------------
# 3. PREPARE PURCHASE HISTORY
# -------------------------------------------------

# Exclude canceled and unavailable orders.
valid_orders = orders[
    (~orders["order_status"].isin(["canceled", "unavailable"])) &
    (orders["order_purchase_timestamp"].notna())
].copy()

purchase_data = valid_orders.merge(
    customers[["customer_id", "customer_unique_id"]],
    on="customer_id",
    how="inner"
)

# Calculate order-level product spending.
order_spending = (
    order_items.groupby("order_id")["price"]
    .sum()
    .reset_index(name="order_value")
)

purchase_data = purchase_data.merge(
    order_spending,
    on="order_id",
    how="left"
)

purchase_data["order_value"] = (
    purchase_data["order_value"].fillna(0)
)

# -------------------------------------------------
# 4. CREATE OBSERVATION AND CHURN PERIODS
# -------------------------------------------------

latest_date = purchase_data["order_purchase_timestamp"].max()

# Use the final 90 days as the label/prediction period.
observation_date = latest_date - pd.Timedelta(days=90)

# Customers must have purchased before the observation date.
historical_data = purchase_data[
    purchase_data["order_purchase_timestamp"] < observation_date
].copy()

future_data = purchase_data[
    (purchase_data["order_purchase_timestamp"] >= observation_date) &
    (purchase_data["order_purchase_timestamp"] <= latest_date)
].copy()

# Customers included in the modelling dataset.
eligible_customers = historical_data[
    "customer_unique_id"
].dropna().unique()

# -------------------------------------------------
# 5. CREATE CUSTOMER FEATURES
# -------------------------------------------------

customer_features = (
    historical_data.groupby("customer_unique_id")
    .agg(
        last_purchase_date=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_value", "sum"),
        average_order_value=("order_value", "mean"),
        first_purchase_date=("order_purchase_timestamp", "min")
    )
    .reset_index()
)

customer_features["recency"] = (
    observation_date - customer_features["last_purchase_date"]
).dt.days

customer_features["customer_tenure"] = (
    customer_features["last_purchase_date"] -
    customer_features["first_purchase_date"]
).dt.days

# -------------------------------------------------
# 6. CREATE CHURN LABEL
# -------------------------------------------------

future_customers = set(
    future_data["customer_unique_id"].dropna().unique()
)

customer_features["churn"] = (
    ~customer_features["customer_unique_id"].isin(future_customers)
).astype(int)

# 1 = No purchase in the next 90 days
# 0 = Made at least one purchase in the next 90 days

customer_features = customer_features.replace(
    [np.inf, -np.inf], np.nan
)

customer_features = customer_features.dropna(
    subset=["recency", "frequency", "monetary", "churn"]
)

customer_features.to_csv(
    OUTPUT_PATH / "customer_churn_dataset.csv",
    index=False
)

print("\n========== CHURN DATASET SUMMARY ==========")
print("Observation date:", observation_date.date())
print("Latest purchase date:", latest_date.date())
print("Eligible customers:", len(customer_features))
print("\nChurn label distribution:")
print(customer_features["churn"].value_counts())

# -------------------------------------------------
# 7. PREPARE MACHINE LEARNING DATA
# -------------------------------------------------

feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "average_order_value",
    "customer_tenure"
]

X = customer_features[feature_columns]
y = customer_features["churn"]

if y.nunique() < 2:
    raise ValueError(
        "Churn dataset contains only one class. "
        "Check the observation period and purchase data."
    )

if y.value_counts().min() < 2:
    raise ValueError(
        "Not enough examples in one churn class for a stratified split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 8. TRAIN MACHINE LEARNING MODELS
# -------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

    trained_models[model_name] = model

    print(f"{model_name} completed.")

# -------------------------------------------------
# 9. COMPARE MODEL PERFORMANCE
# -------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH / "model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Select the model with the highest ROC AUC.
best_model_name = results_df.sort_values(
    "ROC AUC", ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

# -------------------------------------------------
# 10. EVALUATE BEST MODEL
# -------------------------------------------------

best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

print("\n========== BEST MODEL CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    best_predictions,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(y_test, best_predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Churned", "Churned"]
)

display.plot()
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig(CHART_PATH / "confusion_matrix.png")
plt.close()

# -------------------------------------------------
# 11. SAVE CHURN PREDICTIONS
# -------------------------------------------------

prediction_results = customer_features.loc[
    X_test.index,
    ["customer_unique_id"]
].copy()

prediction_results["actual_churn"] = y_test
prediction_results["predicted_churn"] = best_predictions
prediction_results["churn_probability"] = best_probabilities

prediction_results["risk_level"] = pd.cut(
    prediction_results["churn_probability"],
    bins=[-0.001, 0.40, 0.70, 1.001],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

prediction_results.to_csv(
    OUTPUT_PATH / "customer_churn_predictions.csv",
    index=False
)

# -------------------------------------------------
# 12. SAVE TRAINED MODEL
# -------------------------------------------------

joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns
    },
    MODEL_PATH / "customer_churn_model.pkl"
)

# -------------------------------------------------
# 13. MODEL PERFORMANCE CHART
# -------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["ROC AUC"]
)

plt.title("Churn Model ROC AUC Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("ROC AUC Score")
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(CHART_PATH / "model_comparison.png")
plt.close()

# -------------------------------------------------
# 14. COMPLETION MESSAGE
# -------------------------------------------------

print("\n========== CHURN PREDICTION COMPLETED ==========")
print("Customer churn dataset generated.")
print("Machine learning models trained.")
print("Model comparison completed.")
print("Customer churn predictions generated.")
print("Best model saved successfully.")

print("\nReports saved in:", OUTPUT_PATH)
print("Charts saved in:", CHART_PATH)
print("Trained model saved in:", MODEL_PATH)
