
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# STEP 1: PROJECT PATHS
# ==========================================

BASE_DIR = r"C:\Users\suraj\OneDrive\AI_Customer_BI_Project"

DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

OUTPUT_DIR = os.path.join(
    BASE_DIR, "data", "processed",
    "sentiment_analysis_outputs"
)

CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ==========================================
# STEP 2: LOAD REVIEW DATASET
# ==========================================

file_path = os.path.join(
    DATA_DIR, "raw","olist_order_reviews_dataset.csv"
)

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Review dataset not found: {file_path}"
    )

reviews = pd.read_csv(file_path)

print("Review dataset loaded successfully!")
print("Dataset shape:", reviews.shape)
print("\nAvailable columns:")
print(reviews.columns.tolist())

# ==========================================
# STEP 3: CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "review_score",
    "review_comment_message"
]

for column in required_columns:
    if column not in reviews.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

# ==========================================
# STEP 4: CLEAN REVIEW DATA
# ==========================================

reviews["review_score"] = pd.to_numeric(
    reviews["review_score"],
    errors="coerce"
)

reviews = reviews.dropna(subset=["review_score"])

reviews = reviews[
    reviews["review_score"].between(1, 5)
].copy()

reviews["review_comment_message"] = (
    reviews["review_comment_message"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

# ==========================================
# STEP 5: CREATE SENTIMENT LABELS
# ==========================================

def classify_sentiment(score):

    if score <= 2:
        return "Negative"

    elif score == 3:
        return "Neutral"

    else:
        return "Positive"


reviews["sentiment"] = reviews["review_score"].apply(
    classify_sentiment
)

# ==========================================
# STEP 6: PREPARE TEXT DATA
# ==========================================

# Use only reviews with actual text for NLP training

text_reviews = reviews[
    reviews["review_comment_message"].str.len() > 0
].copy()

if len(text_reviews) < 10:
    raise ValueError(
        "Not enough text reviews to train the sentiment model."
    )

X = text_reviews["review_comment_message"]
y = text_reviews["sentiment"]

print("\nSentiment distribution:")
print(y.value_counts())

# ==========================================
# STEP 7: CONVERT TEXT INTO TF-IDF FEATURES
# ==========================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_features = vectorizer.fit_transform(X)

# ==========================================
# STEP 8: SPLIT TRAINING AND TESTING DATA
# ==========================================

class_counts = y.value_counts()

if class_counts.min() < 2:
    raise ValueError(
        "Each sentiment class needs at least two text reviews."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ==========================================
# STEP 9: TRAIN SENTIMENT MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nSentiment model training completed!")

# ==========================================
# STEP 10: EVALUATE MODEL
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# Save evaluation report
report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(OUTPUT_DIR, "sentiment_model_evaluation.csv")
)

# ==========================================
# STEP 11: PREDICT SENTIMENT FOR ALL REVIEWS
# ==========================================

# Reviews without text retain their rating-based
# sentiment label. Text reviews receive model predictions.

reviews["predicted_sentiment"] = reviews["sentiment"]

all_text_mask = (
    reviews["review_comment_message"].str.len() > 0
)

all_text = reviews.loc[
    all_text_mask, "review_comment_message"
]

all_features = vectorizer.transform(all_text)

reviews.loc[
    all_text_mask, "predicted_sentiment"
] = model.predict(all_features)

# ==========================================
# STEP 12: SAVE SENTIMENT RESULTS
# ==========================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "customer_sentiment_analysis.csv"
)

reviews.to_csv(output_csv, index=False)

print("\nSentiment results saved successfully!")

# ==========================================
# STEP 13: SENTIMENT SUMMARY
# ==========================================

sentiment_summary = (
    reviews["predicted_sentiment"]
    .value_counts()
    .reset_index()
)

sentiment_summary.columns = [
    "sentiment",
    "review_count"
]

sentiment_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sentiment_summary.csv"
    ),
    index=False
)

print("\nSentiment Summary:")
print(sentiment_summary.to_string(index=False))

# ==========================================
# STEP 14: GENERATE CHARTS
# ==========================================

sns.set_theme(style="whitegrid")

# Chart 1: Sentiment distribution

plt.figure(figsize=(9, 6))

sns.countplot(
    data=reviews,
    x="predicted_sentiment",
    order=["Positive", "Neutral", "Negative"]
)

plt.title("Customer Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "sentiment_distribution.png"
    )
)

plt.close()

# Chart 2: Review score distribution

plt.figure(figsize=(9, 6))

sns.countplot(
    data=reviews,
    x="review_score",
    order=[1, 2, 3, 4, 5]
)

plt.title("Customer Review Rating Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "review_score_distribution.png"
    )
)

plt.close()

# Chart 3: Confusion matrix

labels = ["Negative", "Neutral", "Positive"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Sentiment Model Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_DIR,
        "sentiment_confusion_matrix.png"
    )
)

plt.close()

# ==========================================
# STEP 15: FINAL OUTPUT
# ==========================================

print("\n====================================")
print("CUSTOMER SENTIMENT ANALYSIS COMPLETED")
print("====================================")

print("\nTotal Reviews:", len(reviews))
print("Reviews Used for NLP:", len(text_reviews))

print("\nOutput Folder:")
print(OUTPUT_DIR)

print("\nCharts Folder:")
print(CHART_DIR)

print("\nGenerated CSV files:")
print("1. customer_sentiment_analysis.csv")
print("2. sentiment_summary.csv")
print("3. sentiment_model_evaluation.csv")

print("\nGenerated Charts:")
print("1. sentiment_distribution.png")
print("2. review_score_distribution.png")
print("3. sentiment_confusion_matrix.png")
