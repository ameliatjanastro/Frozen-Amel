import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------
# Load Data
# ----------------------
base_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/export?format=csv"
urls = {
    "gv": f"{base_url}&gid=0",
    "vendor": f"{base_url}&gid=1841763572",
    "oos": f"{base_url}&gid=1511488791",
}

@st.cache_data
def load_data():
    gv_df = pd.read_csv(urls["gv"])
    vendor_df = pd.read_csv(urls["vendor"])
    oos_df = pd.read_csv(urls["oos"])
    return gv_df, vendor_df, oos_df

gv, vendor, oos = load_data()

# ----------------------
# Preprocessing
# ----------------------
# ----------------------
# Preprocessing
# ----------------------
gv.columns = gv.columns.str.strip()
oos.columns = oos.columns.str.strip()
vendor.columns = vendor.columns.str.strip()

# Convert to numerics
for col in ["goods_value", "quantity_sold"]:
    if col in gv.columns:
        gv[col] = pd.to_numeric(gv[col], errors="coerce").fillna(0)

# Parse dates
if "date_key" in gv.columns:
    gv["date_key"] = pd.to_datetime(gv["date_key"], errors="coerce", dayfirst=True)
if "request_shipping_date: Day" in oos.columns:
    oos["request_shipping_date: Day"] = pd.to_datetime(oos["request_shipping_date: Day"], errors="coerce", dayfirst=True)

# Clean FR
if "FR" in oos.columns:
    oos["FR"] = oos["FR"].astype(str).str.replace('%', '').astype(float)

# Daily Aggregation
daily_agg = gv.groupby("product_id").agg({
    "goods_value": "sum",
    "quantity_sold": "sum",
    "product_name": "first",
    "l1_category_name": "first",
    "l2_category_name": "first",
    "Month": "first"
}).reset_index()

# Merge OOS with product names
oos_merged = pd.merge(oos, gv[["product_id", "product_name"]].drop_duplicates(), on="product_id", how="left")

# Merge GV + OOS for Forecast
merged_gv_oos = pd.merge(
    gv,
    oos,
    left_on=["product_id", "date_key"],
    right_on=["product_id", "request_shipping_date: Day"],
    how="left"
)

# ----------------------
# Streamlit UI Setup
# ----------------------
st.set_page_config(page_title="Frozen SKU Dashboard", layout="wide")
st.title("🧊 Frozen SKU Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Summary", "📈 Trend", "🚨 OOS Risk", "🏦 Vendor Scorecard", "📦 Forecast Accuracy"])

# ----------------------
# Summary Tab
# ----------------------
with tab1:
    st.header("Summary Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total SKUs", gv["product_id"].nunique())
    col2.metric("Total Goods Value", f"Rp {gv['goods_value'].sum():,.0f}")
    col3.metric("Total Qty Sold", int(gv["quantity_sold"].sum()))

    st.subheader("Top 10 SKUs by Quantity Sold")
    top_qty = daily_agg.sort_values("quantity_sold", ascending=False).head(10)
    st.dataframe(top_qty[["product_id", "product_name", "quantity_sold", "goods_value"]])

# ----------------------
# Trend Tab
# ----------------------
with tab2:
    st.header("Sales Trend by Product")
    col1, col2 = st.columns(2)
    selected_vendor = col1.selectbox("Filter by Vendor", sorted(gv["l1_category_name"].dropna().unique()))
    filtered_skus = gv[gv["l1_category_name"] == selected_vendor]
    selected_sku = col2.selectbox("Select SKU", sorted(filtered_skus["product_name"].dropna().unique()))
    sku_data = filtered_skus[filtered_skus["product_name"] == selected_sku].sort_values("date_key")

    fig, ax = plt.subplots()
    ax.plot(sku_data["date_key"], sku_data["quantity_sold"], marker='o')
    ax.set_title(f"Sales Trend: {selected_sku}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity Sold")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ----------------------
# OOS Risk Tab
# ----------------------
with tab3:
    st.header("Out-of-Stock Risk Items")
    threshold = st.slider("Min Qty Threshold", 0, 50, 10)

    if {"SUM of po_qty", "SUM of actual_qty"}.issubset(oos_merged.columns):
        risky = oos_merged[oos_merged["SUM of po_qty"] > oos_merged["SUM of actual_qty"]]
        risky = pd.merge(risky, daily_agg[["product_id", "quantity_sold"]], on="product_id", how="left")
        risky = risky[risky["quantity_sold"] >= threshold]
        st.dataframe(risky[[ "product_id", "product_name", "SUM of po_qty", "SUM of actual_qty", "FR", "quantity_sold"]].dropna())
    else:
        st.warning("OOS sheet missing PO quantity columns.")

# ----------------------
# Vendor Scorecard Tab
# ----------------------
with tab4:
    st.header("Vendor Fulfillment Scorecard")
    if "FR" in oos_merged.columns:
        vendor_score = oos_merged.groupby("product_id").agg({
            "FR": "mean",
            "product_name": "first"
        }).reset_index()
        st.dataframe(vendor_score.sort_values("FR"))
    else:
        st.warning("FR column missing in OOS data.")

# ----------------------
# Forecast Accuracy Tab
# ----------------------
with tab5:
    st.header("Forecast Accuracy (MAPE)")

    if {"Forecast Qty", "Actual Sales (Qty)"}.issubset(merged_gv_oos.columns):
        forecast_df = merged_gv_oos.copy()
        forecast_df["Forecast Qty"] = pd.to_numeric(forecast_df["Forecast Qty"], errors="coerce")
        forecast_df["Actual Sales (Qty)"] = pd.to_numeric(forecast_df["Actual Sales (Qty)"], errors="coerce")

        forecast_df["Error"] = (forecast_df["Forecast Qty"] - forecast_df["Actual Sales (Qty)"]).abs()
        forecast_df["APE"] = forecast_df["Error"] / (forecast_df["Actual Sales (Qty)"] + 1e-9)
        
        mape_df = forecast_df.groupby("product_id").agg({
            "APE": "mean",
            "product_name": "first",
            "FR": "mean"
        }).reset_index()
        mape_df["MAPE"] = mape_df["APE"] * 100

        acc_df = mape_df[["product_id", "product_name", "MAPE", "FR"]].sort_values("MAPE")

        st.dataframe(acc_df.head(15))
        csv = acc_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Forecast Accuracy CSV", csv, file_name="forecast_accuracy_summary.csv", mime="text/csv")
    else:
        st.warning("Forecast or actual sales data not available in the dataset.")

