import streamlit as st
import pandas as pd

# Load the data
base_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/export?format=csv"
urls = {
    "gv": f"{base_url}&gid=0",
    "vendor": f"{base_url}&gid=1841763572",
    "oos": f"{base_url}&gid=1511488791",
    "fr": f"{base_url}&gid=1390643918",  # Assumes FR data added here
}

@st.cache_data
def load_data():
    return {
        k: pd.read_csv(v) for k, v in urls.items()
    }

data = load_data()
gv, vendor, oos, fr = data["gv"], data["vendor"], data["oos"], data["fr"]

# === Convert dates ===
gv["date_key"] = pd.to_datetime(gv["date_key"], errors="coerce")
oos["Date"] = pd.to_datetime(oos["Date"], errors="coerce")

# === Get latest 30 days ===
latest_date = gv["date_key"].max()
cutoff = latest_date - pd.Timedelta(days=30)
gv_l30 = gv[gv["date_key"] >= cutoff]
oos_l30 = oos[oos["Date"] >= cutoff]

# === Streamlit Layout ===
st.set_page_config(layout="wide")
st.title("📦 Frozen SKU Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Summary", "GV Trend", "Vendor Scorecard", "Forecast Accuracy"])

# ===== TAB 1: Summary =====
with tab1:
    st.header("📊 Last 30 Days Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total July Sales", f"{gv_l30['quantity_sold'].sum():,.0f}")
    col2.metric("Total Goods Value", f"Rp {gv_l30['goods_value'].sum():,.0f}")
    col3.metric("Unique SKUs", gv_l30["product_id"].nunique())

    # --- High-Value but OOS-Risk Detection ---
    gv_agg = gv_l30.groupby("product_id").agg({
        "goods_value": "sum",
        "product_name": "first",
        "l1_category_name": "first",
        "l2_category_name": "first"
    }).reset_index()

    oos_agg = oos_l30.groupby("product_id").agg({
        "%OOS Today": "mean",
        "Vendor Name": "first",
        "DOI Hub": "mean",
        "Stock WH": "mean"
    }).reset_index()

    risk_df = pd.merge(gv_agg, oos_agg, on="product_id", how="inner")
    top_quartile = risk_df["goods_value"].quantile(0.75)
    high_risk = risk_df[(risk_df["goods_value"] >= top_quartile) & (risk_df["%OOS Today"] >= 30)]

    st.subheader("🔥 High-Value but OOS-Risk SKUs (L30)")
    if not high_risk.empty:
        st.dataframe(high_risk[[
            "product_id", "product_name", "Vendor Name", "goods_value",
            "%OOS Today", "DOI Hub", "Stock WH"
        ]])
    else:
        st.info("No high-value, high-OOS SKUs in the last 30 days.")

# ===== TAB 2: GV Trend =====
with tab2:
    st.header("📈 GV Trend by Category")
    gv_trend = gv_l30.groupby(["date_key", "l1_category_name"]).agg({
        "goods_value": "sum"
    }).reset_index()
    st.line_chart(
        gv_trend.pivot(index="date_key", columns="l1_category_name", values="goods_value")
    )

# ===== TAB 3: Vendor Scorecard =====
with tab3:
    st.header("🏭 Vendor Scorecard")
    st.dataframe(vendor[[
        "Vendor Name", "Total SKU", "SUM of Request Qty",
        "SUM of Actual Qty", "FR"
    ]].sort_values("FR"))

# ===== TAB 4: Forecast Accuracy =====
with tab4:
    st.header("📉 Forecast Accuracy (L30)")
    forecast_df = oos_l30.copy()
    forecast_df = forecast_df.dropna(subset=["Forecast Qty", "Actual Sales (Qty)"])
    forecast_df["Error"] = abs(forecast_df["Forecast Qty"] - forecast_df["Actual Sales (Qty)"])
    forecast_df["Accuracy"] = 1 - (forecast_df["Error"] / forecast_df["Forecast Qty"].replace(0, 1))

    top_acc = forecast_df.groupby("product_id").agg({
        "Accuracy": "mean",
        "product_name": "first",
        "Vendor Name": "first"
    }).reset_index().sort_values("Accuracy", ascending=False)

    st.dataframe(top_acc.head(30), use_container_width=True)


