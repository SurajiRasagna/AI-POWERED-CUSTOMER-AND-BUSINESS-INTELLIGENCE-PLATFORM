# Web Application Guide

The project now has two ways to explore the intelligence: the original
Streamlit dashboard and a full **Flask web app** with multi-user accounts.

This guide covers the Flask app.

## What it includes

| Area | Description |
|------|-------------|
| Authentication | Register, sign in, sign out, session management |
| Roles | Admin, Analyst, Viewer with server-side enforcement |
| Overview | Revenue trend, top categories, sentiment and demand forecast |
| Sales | Monthly revenue, payments, categories, states, sales forecast |
| Customers | RFM segments, lifetime value, churn risk and churn models |
| Product Demand | Per-product history and the saved three-month forecast |
| Sentiment | NLP sentiment distribution, ratings and latest reviews |
| Live Prediction | Instant single-product scoring and CSV batch scoring |
| Audit log | Every live prediction is stored against the signed-in user |

## Project structure

```
AI_Customer_BI_Project/
├── run_web.py                     # Web app entry point
├── start_web.bat                  # Windows one-click launcher
├── backend/routes/routes/
│   ├── app.py                     # Flask app factory
│   ├── config.py                  # Paths, database, secret key
│   ├── database.py                # SQLAlchemy extension
│   ├── models.py                  # User + PredictionLog
│   ├── security.py                # Sessions, CSRF, role guards
│   ├── auth.py                    # Register / login / logout
│   ├── admin.py                   # User management
│   ├── views.py                   # HTML routes
│   ├── api.py                     # JSON API
│   ├── data_service.py            # Analytics CSV loaders
│   └── ml_service.py              # Model loading and predictions
├── frontend/templates/            # Jinja pages
├── frontend/static/css/style.css  # Design system
├── frontend/static/js/app.js      # API, charts, tables, helpers
├── instance/app.db                # SQLite database (created on first run)
└── models/                        # Trained .pkl models
```

## Run it

```powershell
cd C:\Users\suraj\OneDrive\AI_Customer_BI_Project
.\venv\Scripts\python.exe run_web.py
```

Or double-click `start_web.bat`.

Open <http://127.0.0.1:5000>.

### First run

1. Click **Create an account**.
2. The **first account becomes the admin**.
3. Additional sign-ups join as **analysts**. Admins can promote them to
   admin or restrict them to viewer from **Users** in the sidebar.

### Roles

| Role | Analytics | Live predictions | User management |
|------|-----------|------------------|-----------------|
| Admin | Yes | Yes | Yes |
| Analyst | Yes | Yes | No |
| Viewer | Yes | No (read-only) | No |

## Configuration

Set environment variables before starting the app.

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` to expose on the LAN. |
| `PORT` | `5000` | HTTP port. |
| `FLASK_ENV` | `development` | `production` disables debug mode. |
| `SECRET_KEY` | auto-generated | Session signing key. Use a strong value in production. |
| `DATABASE_URL` | SQLite | `mysql+pymysql://user:pass@host/dbname` for MySQL. |
| `COOKIE_SECURE` | `0` | Set to `1` when serving over HTTPS. |

The auto-generated secret is stored in `instance/secret_key.txt`. Do not
commit that file.

### Example: MySQL

```powershell
$env:DATABASE_URL = "mysql+pymysql://root:password@localhost/customer_bi"
$env:FLASK_ENV = "production"
$env:SECRET_KEY = "replace-with-a-long-random-value"
.\venv\Scripts\python.exe run_web.py
```

Tables are created automatically on startup.

## JSON API

All API routes require a signed-in session. POST requests also require the
`X-CSRF-Token` header (the frontend supplies it automatically).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/overview` | KPIs, monthly sales, top categories, sentiment, forecast |
| GET | `/api/sales` | Sales analytics and sales forecast |
| GET | `/api/customers` | Segments, CLV, churn risk and model scores |
| GET | `/api/demand` | Product demand history, forecast, evaluation |
| GET | `/api/sentiment` | Sentiment distribution, ratings, recent reviews |
| GET | `/api/products` | Trained products and model metadata |
| GET | `/api/ml/info` | Model version, training time, features |
| POST | `/api/predict/demand` | Single-product live prediction |
| POST | `/api/predict/demand/batch` | CSV batch scoring (multipart `file`) |
| GET | `/api/predictions` | Prediction audit log |

### Single prediction example

```json
POST /api/predict/demand
{
  "product_id": "aca2eb7d00ea1a7b8ebd4e68314663af",
  "lag_1": 3,
  "lag_2": 2,
  "lag_3": 2,
  "forecast_month": "2018-09"
}
```

Response:

```json
{
  "prediction": 7.8,
  "signal": "High demand",
  "safety_stock": 11.68,
  "reorder_point": 19.48,
  "forecast": [
    {"forecast_month": "2018-09", "predicted_quantity": 8},
    {"forecast_month": "2018-10", "predicted_quantity": 15},
    {"forecast_month": "2018-11", "predicted_quantity": 37}
  ]
}
```

### Batch CSV

Required columns:

| product_id | month | quantity_sold |
|------------|-------|---------------|
| aca2eb7d… | 2018-06-01 | 2 |
| aca2eb7d… | 2018-07-01 | 2 |
| aca2eb7d… | 2018-08-01 | 3 |

The latest three rows per product are used. Rows are skipped with a clear
status when a product has no trained model or too little history.

## Security notes

- Passwords are hashed with Werkzeug PBKDF2; plain passwords are never stored.
- State-changing requests are protected by per-session CSRF tokens.
- Sessions are HTTP-only, SameSite=Lax and expire after eight hours.
- Role checks run on the server, so hiding buttons is not the only defence.
- Uploads are limited to 4 MB and 5,000 rows per batch.
- The last active admin cannot be demoted or deactivated.
- For production, serve behind HTTPS and set `COOKIE_SECURE=1`.

## Data freshness

The app reads the processed CSVs and `.pkl` model files directly. When new
data arrives:

1. Update `data/processed/*.csv`.
2. Re-run the relevant notebook (`step8`–`step10`).
3. Refresh the page. File caches invalidate on modification time.

No server restart is required for CSV updates. Restart the app if you replace
the model file while it is cached.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Demand model file is missing` | Run `notebooks/step10_product_demand_forecasting.py`. |
| Charts blank | Chart.js loads from a CDN; check internet access. Tables and KPIs still work offline. |
| `Invalid security token` | Refresh the page; the session token may have expired. |
| Cannot sign in | Ask an admin to check that your account is active. |
| database is locked (SQLite) | Stop duplicate app instances, or switch to MySQL. |

## Original Streamlit dashboard

The Streamlit app still works:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Both interfaces use the same models and processed data.
