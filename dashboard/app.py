

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import live_prediction as live

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Business Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(r"C:\Users\suraj\OneDrive\AI_Customer_BI_Project")
DATA_DIR = BASE_DIR / "data" / "processed"

CLV_DIR = DATA_DIR / "customer_lifetime_value_outputs"
SENTIMENT_DIR = DATA_DIR / "sentiment_analysis_outputs"
DEMAND_DIR = DATA_DIR / "product_demand_forecasting_outputs"

MODELS_DIR = BASE_DIR / "models"
DEMAND_MODEL_FILE = MODELS_DIR / "product_demand_forecasting_models.pkl"
DEMAND_TEST_PREDICTIONS = DEMAND_DIR / "product_demand_test_predictions.csv"
DEMAND_BATCH_FORECAST = (
    DEMAND_DIR / "product_demand_forecast_next_3_months.csv"
)

# --------------------------------------------------
# 2. PROFESSIONAL CSS DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 55%, #ecfeff 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101b3d 0%, #172554 55%, #312e81 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.08);
    padding: 12px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    transition: 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(56,189,248,0.28);
}

/* Main header */
.main-header {
    background: linear-gradient(120deg, #172554, #4338ca, #0891b2);
    padding: 30px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px rgba(30, 41, 99, 0.18);
}

.main-header h1 {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin: 0;
}

.main-header p {
    color: #dbeafe;
    margin-top: 10px;
    font-size: 15px;
}

/* Section heading */
.section-heading {
    padding: 17px 22px;
    border-radius: 14px;
    color: white;
    font-size: 23px;
    font-weight: 800;
    margin: 20px 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
}

.section-description {
    color: #475569;
    font-size: 14px;
    margin: 0 0 20px 2px;
}

/* KPI cards */
.kpi-card {
    padding: 22px;
    border-radius: 16px;
    min-height: 125px;
    color: white;
    box-shadow: 0 8px 20px rgba(0,0,0,0.10);
    margin-bottom: 12px;
}

.kpi-card .kpi-icon {
    font-size: 25px;
}

.kpi-card .kpi-label {
    font-size: 13px;
    opacity: 0.9;
    margin-top: 10px;
}

.kpi-card .kpi-value {
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}

/* Chart containers */
[data-testid="stPlotlyChart"] {
    background: white;
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 5px 18px rgba(15,23,42,0.06);
}

/* Tables */
[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
    padding: 8px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 25px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_resource
def load_demand_models(path):
    return live.load_models(path)


orders = load_csv(DATA_DIR / "orders_cleaned.csv")
payments = load_csv(DATA_DIR / "payments_cleaned.csv")

clv = load_csv(CLV_DIR / "customer_lifetime_value_analysis.csv")
sentiment = load_csv(SENTIMENT_DIR / "customer_sentiment_analysis.csv")

demand = load_csv(DEMAND_DIR / "monthly_product_demand.csv")
forecast = load_csv(DEMAND_DIR / "product_demand_forecast_next_3_months.csv")


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def find_column(df, possibilities):
    """Find a matching column without depending on capitalization."""
    if df.empty:
        return None

    lookup = {str(c).lower().strip(): c for c in df.columns}

    for name in possibilities:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def money(value):
    try:
        return "R$ {:,.2f}".format(float(value))
    except (ValueError, TypeError):
        return "R$ 0.00"


def show_kpi(label, value, icon, gradient):
    st.markdown(f"""
    <div class="kpi-card" style="background:{gradient};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, description, gradient):
    st.markdown(
        f'<div class="section-heading" style="background:{gradient};">'
        f'{title}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="section-description">{description}</div>',
        unsafe_allow_html=True
    )


def style_chart(fig, title=None):
    fig.update_layout(
        title=title,
        template="plotly_white",
        font=dict(family="Inter", color="#334155"),
        title_font=dict(size=18, color="#172554"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=25, r=25, t=65, b=35),
        legend=dict(orientation="h", y=-0.2)
    )
    return fig


# --------------------------------------------------
# 5. PREPARE SALES DATA
# --------------------------------------------------

sales_data = pd.DataFrame()

if not orders.empty and not payments.empty:
    order_id_col = find_column(orders, ["order_id"])
    payment_order_col = find_column(payments, ["order_id"])
    payment_value_col = find_column(payments, ["payment_value"])

    if order_id_col and payment_order_col and payment_value_col:
        payment_summary = (
            payments.groupby(payment_order_col)[payment_value_col]
            .sum()
            .reset_index()
        )

        payment_summary.columns = ["order_id", "revenue"]

        sales_data = orders.merge(
            payment_summary,
            left_on=order_id_col,
            right_on="order_id",
            how="inner"
        )

        status_col = find_column(sales_data, ["order_status"])

        if status_col:
            sales_data = sales_data[
                ~sales_data[status_col].astype(str).str.lower().isin(
                    ["canceled", "unavailable"]
                )
            ].copy()

        date_col = find_column(
            sales_data,
            ["order_purchase_timestamp", "purchase_date", "order_date"]
        )

        if date_col:
            sales_data["dashboard_date"] = pd.to_datetime(
                sales_data[date_col], errors="coerce"
            )
            sales_data["month"] = (
                sales_data["dashboard_date"].dt.to_period("M").astype(str)
            )


# --------------------------------------------------
# 6. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 📊 AI INSIGHTS")
    st.markdown("### Business Intelligence")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Business Overview",
            "💰 Sales Analysis",
            "👥 Customer Analysis",
            "📦 Product Demand",
            "🔮 Live Demand Prediction",
            "💬 Sentiment Analysis"
        ]
    )

    st.markdown("---")
    st.markdown("### 🤖 Project Information")
    st.caption("AI-Powered Customer & Business Intelligence Platform")
    st.caption("Python • Machine Learning • Streamlit")
    st.markdown("---")
    st.caption("Dashboard Version 2.0")


# --------------------------------------------------
# 7. MAIN HEADER
# --------------------------------------------------

st.markdown("""
<div class="main-header">
    <h1>🚀 AI-Powered Customer & Business Intelligence</h1>
    <p>Transforming e-commerce data into meaningful business decisions
    through analytics, forecasting and artificial intelligence.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# PAGE 1: BUSINESS OVERVIEW
# --------------------------------------------------

if page == "🏠 Business Overview":

    section_header(
        "🏠 Business Overview",
        "A complete snapshot of your business performance and customer activity.",
        "linear-gradient(100deg,#2563eb,#4f46e5)"
    )

    total_revenue = (
        sales_data["revenue"].sum() if not sales_data.empty else 0
    )
    total_orders = (
        sales_data["order_id"].nunique() if not sales_data.empty else 0
    )

    customer_col = find_column(clv, ["customer_unique_id", "customer_id"])
    total_customers = (
        clv[customer_col].nunique()
        if not clv.empty and customer_col else 0
    )

    avg_order_value = (
        total_revenue / total_orders if total_orders else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_kpi("Total Revenue", money(total_revenue), "💰",
                 "linear-gradient(135deg,#2563eb,#1d4ed8)")
    with c2:
        show_kpi("Total Orders", f"{total_orders:,}", "🛒",
                 "linear-gradient(135deg,#0891b2,#0e7490)")
    with c3:
        show_kpi("Customers Analyzed", f"{total_customers:,}", "👥",
                 "linear-gradient(135deg,#7c3aed,#6d28d9)")
    with c4:
        show_kpi("Average Order Value", money(avg_order_value), "📈",
                 "linear-gradient(135deg,#059669,#047857)")

    st.markdown("### 📌 Business Highlights")

    left, right = st.columns(2)

    with left:
        st.info(
            "💡 **Revenue Insight:** Use the Sales Analysis section "
            "to explore monthly revenue and order performance."
        )

    with right:
        st.success(
            "🎯 **Customer Insight:** Explore customer value segments "
            "and sentiment to understand customer behaviour."
        )

    if not sales_data.empty and "month" in sales_data.columns:
        monthly = (
            sales_data.groupby("month")["revenue"]
            .sum().reset_index()
        )

        fig = px.area(
            monthly,
            x="month",
            y="revenue",
            markers=True,
            color_discrete_sequence=["#4f46e5"]
        )
        fig.update_traces(fill="tozeroy")
        st.plotly_chart(
            style_chart(fig, "📈 Monthly Revenue Trend"),
            use_container_width=True
        )
    else:
        st.warning("Sales chart unavailable. Check the orders and payments CSV files.")


# --------------------------------------------------
# PAGE 2: SALES ANALYSIS
# --------------------------------------------------

elif page == "💰 Sales Analysis":

    section_header(
        "💰 Sales Analytics",
        "Understand revenue trends, order performance and sales patterns.",
        "linear-gradient(100deg,#059669,#0d9488)"
    )

    if not sales_data.empty:

        c1, c2, c3 = st.columns(3)

        with c1:
            show_kpi("Total Revenue", money(sales_data["revenue"].sum()),
                     "💵", "linear-gradient(135deg,#059669,#047857)")

        with c2:
            show_kpi("Total Transactions",
                     f"{sales_data['order_id'].nunique():,}",
                     "🧾", "linear-gradient(135deg,#0d9488,#0f766e)")

        with c3:
            avg = sales_data.groupby("order_id")["revenue"].sum().mean()
            show_kpi("Average Revenue / Order", money(avg),
                     "📊", "linear-gradient(135deg,#0891b2,#0369a1)")

        if "month" in sales_data.columns:
            monthly = (
                sales_data.groupby("month")
                .agg(
                    Revenue=("revenue", "sum"),
                    Orders=("order_id", "nunique")
                ).reset_index()
            )

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    monthly, x="month", y="Revenue",
                    color="Revenue",
                    color_continuous_scale="Teal"
                )
                st.plotly_chart(
                    style_chart(fig, "Monthly Revenue"),
                    use_container_width=True
                )

            with col2:
                fig = px.line(
                    monthly, x="month", y="Orders",
                    markers=True,
                    color_discrete_sequence=["#f97316"]
                )
                st.plotly_chart(
                    style_chart(fig, "Monthly Order Count"),
                    use_container_width=True
                )

            st.markdown("### 📋 Monthly Sales Summary")
            st.dataframe(monthly, use_container_width=True)

        else:
            st.warning("A valid order date column was not found.")

    else:
        st.warning("Sales data is unavailable. Check your cleaned orders and payments files.")


# --------------------------------------------------
# PAGE 3: CUSTOMER ANALYSIS
# --------------------------------------------------

elif page == "👥 Customer Analysis":

    section_header(
        "👥 Customer Intelligence",
        "Analyze customer lifetime value, spending behaviour and customer segments.",
        "linear-gradient(100deg,#7c3aed,#9333ea)"
    )

    if not clv.empty:

        clv_value_col = find_column(
            clv,
            ["historical_clv", "customer_lifetime_value", "clv",
             "total_spending", "total_payment_value"]
        )

        segment_col = find_column(
            clv,
            ["customer_value_segment", "value_segment", "segment",
             "customer_segment"]
        )

        if clv_value_col:
            total_clv = pd.to_numeric(
                clv[clv_value_col], errors="coerce"
            ).fillna(0).sum()

            avg_clv = pd.to_numeric(
                clv[clv_value_col], errors="coerce"
            ).mean()

            c1, c2, c3 = st.columns(3)

            with c1:
                show_kpi("Customers", f"{len(clv):,}", "👤",
                         "linear-gradient(135deg,#7c3aed,#6d28d9)")

            with c2:
                show_kpi("Total Historical CLV", money(total_clv), "💎",
                         "linear-gradient(135deg,#9333ea,#7e22ce)")

            with c3:
                show_kpi("Average Customer Value", money(avg_clv), "⭐",
                         "linear-gradient(135deg,#db2777,#be185d)")

            if segment_col:
                segment_summary = (
                    clv[segment_col].fillna("Unknown")
                    .value_counts().reset_index()
                )
                segment_summary.columns = ["Segment", "Customers"]

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.pie(
                        segment_summary,
                        names="Segment",
                        values="Customers",
                        hole=0.48,
                        color_discrete_sequence=[
                            "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981"
                        ]
                    )
                    st.plotly_chart(
                        style_chart(fig, "Customer Segment Distribution"),
                        use_container_width=True
                    )

                with col2:
                    fig = px.bar(
                        segment_summary,
                        x="Segment",
                        y="Customers",
                        color="Segment",
                        color_discrete_sequence=[
                            "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981"
                        ]
                    )
                    st.plotly_chart(
                        style_chart(fig, "Customers by Segment"),
                        use_container_width=True
                    )

            customer_id_col = find_column(
                clv, ["customer_unique_id", "customer_id"]
            )

            if customer_id_col:
                top = clv.nlargest(10, clv_value_col)

                fig = px.bar(
                    top,
                    x=customer_id_col,
                    y=clv_value_col,
                    color=clv_value_col,
                    color_continuous_scale="Purples"
                )
                st.plotly_chart(
                    style_chart(fig, "Top 10 High-Value Customers"),
                    use_container_width=True
                )

            st.markdown("### 📋 Customer Analysis Data")
            st.dataframe(clv.head(100), use_container_width=True)

        else:
            st.warning(
                "Could not identify the customer value column. "
                "Available columns: " + ", ".join(clv.columns.astype(str))
            )

    else:
        st.warning("Customer CLV file is unavailable. Check Step 11 outputs.")


# --------------------------------------------------
# PAGE 4: PRODUCT DEMAND
# --------------------------------------------------

elif page == "📦 Product Demand":

    section_header(
        "📦 Product Demand Forecasting",
        "Explore historical product demand and the next three months of forecast results.",
        "linear-gradient(100deg,#ea580c,#f59e0b)"
    )

    if not demand.empty:

        st.markdown("### 📊 Historical Product Demand")

        st.dataframe(demand.head(100), use_container_width=True)

        numeric_cols = demand.select_dtypes(include="number").columns.tolist()

        product_col = find_column(
            demand, ["product_id", "product_category_name"]
        )
        demand_value_col = find_column(
            demand, ["quantity_sold", "demand", "order_count",
                     "total_demand", "sales"]
        )
        month_col = find_column(
            demand, ["month", "order_month", "year_month"]
        )

        if demand_value_col and month_col:
            fig = px.line(
                demand,
                x=month_col,
                y=demand_value_col,
                markers=True,
                color_discrete_sequence=["#f97316"]
            )
            st.plotly_chart(
                style_chart(fig, "Historical Demand Trend"),
                use_container_width=True
            )

        elif product_col and demand_value_col:
            product_summary = (
                demand.groupby(product_col)[demand_value_col]
                .sum().nlargest(10).reset_index()
            )

            fig = px.bar(
                product_summary,
                x=product_col,
                y=demand_value_col,
                color=demand_value_col,
                color_continuous_scale="Oranges"
            )
            st.plotly_chart(
                style_chart(fig, "Top 10 Products by Demand"),
                use_container_width=True
            )

        elif numeric_cols:
            st.info(
                "Your demand data is loaded. Check the table above "
                "for its available columns."
            )

    else:
        st.warning("Historical demand file is unavailable. Check Step 10 outputs.")

    st.markdown("### 🔮 Future Demand Forecast")

    if not forecast.empty:
        st.dataframe(forecast, use_container_width=True)

        forecast_numeric = forecast.select_dtypes(
            include="number"
        ).columns.tolist()

        if forecast_numeric:
            forecast_col = forecast_numeric[-1]
            fig = px.bar(
                forecast,
                y=forecast_col,
                color=forecast_col,
                color_continuous_scale="Sunset"
            )
            st.plotly_chart(
                style_chart(fig, "Forecasted Product Demand"),
                use_container_width=True
            )
    else:
        st.warning("Forecast file is unavailable. Check Step 10 outputs.")


# --------------------------------------------------
# PAGE 5: LIVE DEMAND PREDICTION
# --------------------------------------------------

elif page == "🔮 Live Demand Prediction":

    section_header(
        "🔮 Live Demand Prediction",
        "Feed current demand figures into the trained per-product models and "
        "get an instant forecast for operations and inventory planning.",
        "linear-gradient(100deg,#0f766e,#0ea5e9)"
    )

    if not DEMAND_MODEL_FILE.exists():
        st.error(
            "Trained demand models were not found at "
            f"`{DEMAND_MODEL_FILE}`. Run "
            "`notebooks/step10_product_demand_forecasting.py` first."
        )

    elif demand.empty:
        st.error(
            "Monthly demand history is unavailable. Check the Step 10 outputs."
        )

    else:
        models = load_demand_models(str(DEMAND_MODEL_FILE))
        history = live.normalize_history(demand)
        products_df = load_csv(DATA_DIR / "products_cleaned.csv")

        trained_at = pd.Timestamp(
            DEMAND_MODEL_FILE.stat().st_mtime, unit="s"
        )
        st.caption(
            f"Model file last trained: {trained_at:%d %b %Y %H:%M} · "
            f"Data through: {history['month'].max():%b %Y}"
        )

        options = live.product_options(history, products_df)
        available = options[options["product_id"].isin(models.keys())]

        if available.empty:
            st.warning(
                "No product in the demand history has a trained model. "
                "Re-run Step 10 to regenerate the model file."
            )

        else:
            label_map = dict(
                zip(available["product_id"], available["label"])
            )

            st.markdown("### 🛒 Select a product")
            selected_id = st.selectbox(
                "Product",
                available["product_id"].tolist(),
                format_func=lambda pid: label_map.get(pid, pid),
                help="Only products with a trained demand model are listed."
            )

            product_hist = live.get_product_history(history, selected_id)
            recent = product_hist.tail(3)

            if len(recent) < 3:
                st.warning(
                    "This product has fewer than three months of history, "
                    "so a live prediction cannot be built."
                )

            else:
                values = recent["quantity_sold"].astype(float).tolist()
                lag_3, lag_2, lag_1 = values[0], values[1], values[2]
                last_month = product_hist["month"].max()
                last_actual = recent["month"].max()

                st.markdown("### 🧾 Current demand inputs")
                st.caption(
                    "Values are pre-filled from the latest months in the "
                    "dataset. Override them with the newest real numbers "
                    "before predicting."
                )

                lag_col_1, lag_col_2, lag_col_3 = st.columns(3)

                with lag_col_1:
                    input_lag_3 = st.number_input(
                        "3 months ago — lag_3",
                        min_value=0.0,
                        value=float(lag_3),
                        step=1.0,
                        key=f"live_lag3_{selected_id}"
                    )

                with lag_col_2:
                    input_lag_2 = st.number_input(
                        "2 months ago — lag_2",
                        min_value=0.0,
                        value=float(lag_2),
                        step=1.0,
                        key=f"live_lag2_{selected_id}"
                    )

                with lag_col_3:
                    input_lag_1 = st.number_input(
                        "Latest month — lag_1",
                        min_value=0.0,
                        value=float(lag_1),
                        step=1.0,
                        key=f"live_lag1_{selected_id}"
                    )

                forecast_months = [
                    last_month + pd.offsets.MonthBegin(step)
                    for step in range(1, 7)
                ]
                month_labels = [
                    m.strftime("%B %Y") for m in forecast_months
                ]

                forecast_col_1, forecast_col_2 = st.columns(2)

                with forecast_col_1:
                    month_label = st.selectbox(
                        "Forecast month",
                        month_labels,
                        index=0,
                        key=f"live_month_{selected_id}"
                    )
                    forecast_month = forecast_months[
                        month_labels.index(month_label)
                    ]

                with forecast_col_2:
                    st.metric(
                        "Latest reported month",
                        last_actual.strftime("%B %Y")
                    )

                default_time_index = int(len(product_hist))

                with st.expander("⚙️ Advanced model inputs"):
                    time_index = st.number_input(
                        "time_index (months in the product's series)",
                        min_value=0,
                        value=default_time_index,
                        step=1,
                        key=f"live_ti_{selected_id}",
                        help=(
                            "The model was trained with time_index equal to "
                            "the product's position in its monthly series."
                        )
                    )

                    st.caption(
                        "Features used by the model: "
                        + ", ".join(live.FEATURE_COLUMNS)
                    )

                current_signature = (
                    selected_id,
                    round(input_lag_1, 4),
                    round(input_lag_2, 4),
                    round(input_lag_3, 4),
                    str(forecast_month.date()),
                    int(time_index),
                )

                st.markdown("---")

                if st.button(
                    "🚀 Predict demand",
                    type="primary",
                    use_container_width=True
                ):
                    try:
                        prediction = live.predict_demand(
                            models,
                            selected_id,
                            [input_lag_3, input_lag_2, input_lag_1],
                            forecast_month,
                            time_index=int(time_index)
                        )

                        forecast = live.forecast_next_months(
                            models,
                            selected_id,
                            [input_lag_3, input_lag_2, input_lag_1],
                            forecast_month,
                            horizon=3,
                            time_index_start=int(time_index)
                        )

                        signal_label, signal_ratio = live.demand_signal(
                            prediction,
                            [input_lag_3, input_lag_2, input_lag_1]
                        )

                        test_predictions = load_csv(DEMAND_TEST_PREDICTIONS)
                        safety, error_std = live.safety_stock(
                            test_predictions,
                            selected_id
                        )

                        st.session_state["live_signature"] = current_signature
                        st.session_state["live_result"] = {
                            "prediction": prediction,
                            "forecast": forecast,
                            "signal": signal_label,
                            "ratio": signal_ratio,
                            "safety": safety,
                            "error_std": error_std,
                        }

                    except Exception as exc:  # noqa: BLE001
                        st.session_state.pop("live_result", None)
                        st.session_state.pop("live_signature", None)
                        st.error(f"Prediction failed: {exc}")

                result = st.session_state.get("live_result")
                result_signature = st.session_state.get("live_signature")

                if result and result_signature == current_signature:
                    prediction = result["prediction"]
                    forecast = result["forecast"]
                    safety = result["safety"]
                    reorder_point = prediction + safety

                    st.markdown("### 📈 Live prediction")

                    k1, k2, k3, k4 = st.columns(4)

                    with k1:
                        show_kpi(
                            f"Demand — {month_label}",
                            f"{prediction:,.0f} units",
                            "📦",
                            "linear-gradient(135deg,#0f766e,#115e59)"
                        )

                    with k2:
                        show_kpi(
                            "Demand signal",
                            result["signal"],
                            "🎯",
                            "linear-gradient(135deg,#2563eb,#1d4ed8)"
                        )

                    with k3:
                        show_kpi(
                            "Safety stock (95%)",
                            f"{safety:,.0f} units",
                            "🛡️",
                            "linear-gradient(135deg,#d97706,#b45309)"
                        )

                    with k4:
                        show_kpi(
                            "Reorder point",
                            f"{reorder_point:,.0f} units",
                            "📊",
                            "linear-gradient(135deg,#7c3aed,#6d28d9)"
                        )

                    if result["error_std"] > 0:
                        st.caption(
                            "Safety stock = 1.65 × standard deviation of the "
                            f"model's test-period error ({result['error_std']:.2f} "
                            "units). Reorder point = forecast + safety stock."
                        )
                    else:
                        st.caption(
                            "Safety stock could not be estimated for this "
                            "product; reorder point equals the forecast."
                        )

                    forecast_depth = st.slider(
                        "Forecast horizon (months)",
                        min_value=1,
                        max_value=3,
                        value=len(forecast),
                        key=f"live_horizon_{selected_id}"
                    )

                    forecast_view = forecast.head(forecast_depth)

                    st.markdown("### 🔮 Next months")

                    st.dataframe(
                        forecast_view.assign(
                            forecast_month=lambda df: (
                                df["forecast_month"].dt.strftime("%Y-%m")
                            ),
                            predicted_quantity=lambda df: (
                                df["predicted_quantity"].astype(int)
                            )
                        )[["forecast_month", "predicted_quantity"]],
                        use_container_width=True,
                        hide_index=True
                    )

                    plot_hist = product_hist.tail(12).copy()
                    plot_hist["actual"] = plot_hist["quantity_sold"]

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=plot_hist["month"],
                        y=plot_hist["actual"],
                        mode="lines+markers",
                        name="Actual demand",
                        line=dict(color="#0f766e", width=3)
                    ))

                    fig.add_trace(go.Scatter(
                        x=forecast_view["forecast_month"],
                        y=forecast_view["predicted_quantity"],
                        mode="lines+markers",
                        name="Live forecast",
                        line=dict(color="#f97316", width=3, dash="dash")
                    ))

                    st.plotly_chart(
                        style_chart(
                            fig,
                            f"Demand outlook — {label_map.get(selected_id, selected_id)}"
                        ),
                        use_container_width=True
                    )

                    saved_forecast = load_csv(DEMAND_BATCH_FORECAST)

                    if not saved_forecast.empty:
                        saved_rows = saved_forecast[
                            saved_forecast["product_id"] == selected_id
                        ]

                        if not saved_rows.empty:
                            with st.expander(
                                "Compare with the batch forecast from Step 10"
                            ):
                                st.dataframe(
                                    saved_rows,
                                    use_container_width=True,
                                    hide_index=True
                                )

                elif result:
                    st.info(
                        "Inputs changed. Click **Predict demand** to refresh "
                        "the result."
                    )

                # ------------------------------------------
                # Batch scoring for live files
                # ------------------------------------------

                st.markdown("---")
                st.markdown("### 📥 Score a live demand file")
                st.caption(
                    "Upload a CSV with `product_id`, `month` and "
                    "`quantity_sold` columns. The latest three rows per "
                    "product are used to predict the following month."
                )

                template = pd.DataFrame({
                    "product_id": [selected_id],
                    "month": [last_actual.strftime("%Y-%m-%d")],
                    "quantity_sold": [int(input_lag_1)],
                })

                st.download_button(
                    "Download CSV template",
                    data=template.to_csv(index=False),
                    file_name="live_demand_template.csv",
                    mime="text/csv"
                )

                uploaded = st.file_uploader(
                    "Upload live demand records",
                    type=["csv"],
                    key="live_batch_upload"
                )

                if uploaded is not None:
                    try:
                        batch_input = pd.read_csv(uploaded)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not read the uploaded CSV: {exc}")
                        batch_input = None

                    batch_source = (uploaded.name, uploaded.size)

                    if batch_input is not None and st.button(
                        "Score uploaded file",
                        use_container_width=True
                    ):
                        try:
                            st.session_state["live_batch_result"] = (
                                live.batch_forecast(
                                    models,
                                    batch_input,
                                    horizon=1,
                                    reference_history=history
                                )
                            )
                            st.session_state["live_batch_source"] = batch_source

                        except Exception as exc:  # noqa: BLE001
                            st.session_state.pop("live_batch_result", None)
                            st.session_state.pop("live_batch_source", None)
                            st.error(f"Batch scoring failed: {exc}")

                    batch_result = st.session_state.get("live_batch_result")
                    stored_source = st.session_state.get("live_batch_source")

                    if batch_result is not None and stored_source == batch_source:
                        st.dataframe(
                            batch_result,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download scored results",
                            data=batch_result.to_csv(index=False),
                            file_name="live_demand_predictions.csv",
                            mime="text/csv"
                        )

                        ok_rows = batch_result[
                            batch_result["status"] == "ok"
                        ]

                        if not ok_rows.empty:
                            st.success(
                                f"Scored {len(ok_rows)} product(s). "
                                f"Total predicted units: "
                                f"{int(ok_rows['predicted_quantity'].sum()):,}."
                            )

                st.caption(
                    f"Model: RandomForestRegressor · "
                    f"{len(models)} trained products · "
                    "features: " + ", ".join(live.FEATURE_COLUMNS)
                )


# --------------------------------------------------
# PAGE 6: SENTIMENT ANALYSIS
# --------------------------------------------------

elif page == "💬 Sentiment Analysis":

    section_header(
        "💬 Customer Sentiment Intelligence",
        "Understand customer opinions through review text and sentiment classification.",
        "linear-gradient(100deg,#db2777,#9333ea)"
    )

    if not sentiment.empty:

        sentiment_col = find_column(
            sentiment,
            ["sentiment", "predicted_sentiment", "sentiment_label",
             "customer_sentiment"]
        )

        review_score_col = find_column(
            sentiment,
            ["review_score", "score", "rating"]
        )

        if sentiment_col:

            sentiment_counts = (
                sentiment[sentiment_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .reset_index()
            )
            sentiment_counts.columns = ["Sentiment", "Reviews"]

            positive = sentiment_counts[
                sentiment_counts["Sentiment"].str.lower().str.contains(
                    "positive", na=False
                )
            ]["Reviews"].sum()

            negative = sentiment_counts[
                sentiment_counts["Sentiment"].str.lower().str.contains(
                    "negative", na=False
                )
            ]["Reviews"].sum()

            neutral = sentiment_counts[
                sentiment_counts["Sentiment"].str.lower().str.contains(
                    "neutral", na=False
                )
            ]["Reviews"].sum()

            c1, c2, c3 = st.columns(3)

            with c1:
                show_kpi("Positive Reviews", f"{positive:,}", "😊",
                         "linear-gradient(135deg,#059669,#047857)")

            with c2:
                show_kpi("Neutral Reviews", f"{neutral:,}", "😐",
                         "linear-gradient(135deg,#d97706,#b45309)")

            with c3:
                show_kpi("Negative Reviews", f"{negative:,}", "😞",
                         "linear-gradient(135deg,#e11d48,#be123c)")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    sentiment_counts,
                    names="Sentiment",
                    values="Reviews",
                    hole=0.5,
                    color_discrete_sequence=[
                        "#10b981", "#f59e0b", "#f43f5e", "#94a3b8"
                    ]
                )
                st.plotly_chart(
                    style_chart(fig, "Customer Sentiment Distribution"),
                    use_container_width=True
                )

            with col2:
                fig = px.bar(
                    sentiment_counts,
                    x="Sentiment",
                    y="Reviews",
                    color="Sentiment",
                    color_discrete_sequence=[
                        "#10b981", "#f59e0b", "#f43f5e", "#94a3b8"
                    ]
                )
                st.plotly_chart(
                    style_chart(fig, "Sentiment Comparison"),
                    use_container_width=True
                )

        else:
            st.warning(
                "Sentiment column not found. Available columns: "
                + ", ".join(sentiment.columns.astype(str))
            )

        if review_score_col:
            score_counts = (
                sentiment[review_score_col]
                .value_counts().sort_index().reset_index()
            )
            score_counts.columns = ["Review Score", "Reviews"]

            fig = px.bar(
                score_counts,
                x="Review Score",
                y="Reviews",
                color="Review Score",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(
                style_chart(fig, "Customer Review Rating Distribution"),
                use_container_width=True
            )

        st.markdown("### 📝 Customer Review Data")
        st.dataframe(sentiment.head(100), use_container_width=True)

    else:
        st.warning("Sentiment analysis file is unavailable. Check Step 12 outputs.")


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    <hr>
    <b>AI-Powered Customer & Business Intelligence Platform</b><br>
    Built with Python, Streamlit, Pandas, Plotly and Machine Learning<br>
    © 2026 | Data-Driven Business Intelligence Dashboard
</div>
""", unsafe_allow_html=True)