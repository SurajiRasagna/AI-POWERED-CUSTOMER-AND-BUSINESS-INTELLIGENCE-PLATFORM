# AI-POWERED-CUSTOMER-AND-BUSINESS-INTELLIGENCE-PLATFORM
## 1. Project Overview

The AI-Powered Customer & Business Intelligence Platform is an integrated analytics system designed to help e-commerce businesses understand their customers, monitor business performance, analyze sales, forecast product demand, and evaluate customer feedback.

The platform uses Python, data analytics, machine learning techniques, and an interactive Streamlit dashboard to convert raw business data into meaningful insights.

## 2. Problem Statement

E-commerce businesses generate large amounts of data from customer transactions, orders, products, payments, and customer reviews.

Analyzing these datasets separately makes it difficult to understand customer behaviour, identify business trends, predict future demand, and make informed decisions.

This project aims to provide a centralized platform for analyzing business data and presenting useful insights.

## 3. Project Objectives

- Collect and integrate business datasets.
- Clean and preprocess raw data.
- Perform exploratory data analysis.
- Analyze sales and revenue trends.
- Understand customer purchasing behaviour.
- Perform customer segmentation and RFM analysis.
- Analyze Customer Lifetime Value (CLV).
- Perform customer churn analysis, if implemented.
- Forecast product demand.
- Analyze customer review sentiment.
- Develop an interactive business intelligence dashboard.

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- Jupyter Notebook

## 5. Main Project Modules

1. Data Collection and Integration
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Sales Analysis
5. Customer Segmentation
6. Customer Lifetime Value Analysis
7. Product Demand Forecasting
8. Customer Sentiment Analysis
9. Interactive Dashboard

## 6. Dashboard Features

The dashboard contains five main sections:

- Business Overview
- Sales Analysis
- Customer Analysis
- Product Demand Forecasting
- Customer Sentiment Analysis

## 7. Expected Outcomes

- Better understanding of business performance.
- Identification of customer value segments.
- Insights into customer purchasing patterns.
- Understanding of customer feedback.
- Product demand forecasting insights.
- Interactive visualization of business information.

## 8. How to Run the Project

## Install required libraries

pip install pandas numpy streamlit plotly scikit-learn

### Run the dashboard

python -m streamlit run dashboard/app.py

## 9. Conclusion

The AI-Powered Customer & Business Intelligence Platform provides an integrated approach to business data analysis.

It helps transform raw e-commerce data into meaningful visual insights that can support data-driven business decisions.

## 10. Future Enhancements

- Real-time business data integration.
- Advanced customer churn prediction.
- Improved forecasting models.
- Automated business reports.
- Deployment to a cloud platform.

## 11. Deploy to Render

The repository includes a Render Blueprint (`render.yaml`) and a slim
`requirements-render.txt` for hosting the Flask web app as a managed service.

### One-click Blueprint deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SurajiRasagna/AI-POWERED-CUSTOMER-AND-BUSINESS-INTELLIGENCE-PLATFORM)

1. Sign in to Render and approve the Blueprint.
2. Render creates the `ai-customer-bi` web service plus a free Postgres
   database, then builds and deploys automatically.
3. Open the service URL and click **Register**. The first account created
   becomes the admin.

### Manual setup

Create a **Web Service** from this repository and use:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build command | `pip install -r requirements-render.txt` |
| Start command | `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 run_web:app` |
| Health check path | `/healthz` |

Environment variables:

- `FLASK_ENV=production`
- `SECRET_KEY` — a long random string (Render can generate one)
- `COOKIE_SECURE=1`
- `DATABASE_URL` — a Postgres connection string (recommended, see below)

### Free-plan caveats

- Free web services have 512 MB RAM and sleep after 15 minutes without
  traffic; the next request takes about a minute to wake the app.
- Free services have an ephemeral filesystem. Without `DATABASE_URL` the app
  falls back to SQLite and **user accounts are lost on every deploy or
  restart**, which is why the Blueprint provisions Postgres.
- Free Render Postgres databases **expire 30 days after creation**. Upgrade
  the database to a paid plan, or recreate it and update `DATABASE_URL`,
  before then.
- If the service runs out of memory while loading the larger sentiment CSV,
  move it to a larger instance plan.

