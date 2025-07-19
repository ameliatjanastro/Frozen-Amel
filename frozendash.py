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

gv['goods_value'] = pd.to_numeric(gv['goods_value'], errors='coerce')
gv['quantity_sold'] = pd.to_numeric(gv['quantity_sold'], errors='coerce')


# Ensure goods_value is numeric
if "goods_value" in gv.columns:
    gv["goods_value"] = pd.to_numeric(gv["goods_value"], errors="coerce").fillna(0)
else:
    gv["goods_value"] = 0

if "quantity_sold" in gv.columns:
    gv["quantity_sold"] = pd.to_numeric(gv["quantity_sold"], errors="coerce").fillna(0)
else:
    gv["quantity_sold"] = 0

gv["date_key"] = pd.to_datetime(gv["date_key"], dayfirst=True, errors="coerce")

daily_agg = gv.groupby("product_id").agg({
    "goods_value": "sum",
    "quantity_sold": "sum",
    "product_name": "first",
    "l1_category_name": "first",
    "l2_category_name": "first",
    "Month": "first"
}).reset_index()

# Merge OOS with GV to get product_name and value
if "product_name" in gv.columns:
    oos_merged = pd.merge(oos, gv[["product_id", "product_name"]].drop_duplicates(), on="product_id", how="left")
else:
    oos_merged = oos.copy()
    oos_merged["product_name"] = "Unknown"

if "FR" in oos_merged.columns:
    oos_merged["FR"] = oos_merged["FR"].str.replace('%','').astype(float)

# ----------------------
# Streamlit UI
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

    st.subheader("Top 10 SKUs by Quantity")
    top_qty = daily_agg.sort_values("quantity_sold", ascending=False).head(10)
    st.dataframe(top_qty[["product_id", "product_name", "quantity_sold", "goods_value"]])

# ----------------------
# Trend Tab
# ----------------------
with tab2:
    st.header("Sales Trend by Product")
    selected_sku = st.selectbox("Select SKU to View Trend", gv["product_name"].unique())
    sku_data = gv[gv["product_name"] == selected_sku].sort_values("date_key")
    fig, ax = plt.subplots()
    ax.plot(sku_data["date_key"], sku_data["quantity_sold"], marker='o')
    ax.set_title(f"Sales Trend: {selected_sku}")
    ax.set_ylabel("Qty Sold")
    ax.set_xlabel("Date")
    st.pyplot(fig)

# ----------------------
# OOS Risk Tab
# ----------------------
with tab3:
    st.header("Out-of-Stock Risk Items")
    threshold = st.slider("Min Qty Threshold", 0, 50, 10)
    high_value = daily_agg[daily_agg["quantity_sold"] > 0]
    if "SUM of po_qty" in oos_merged.columns and "SUM of actual_qty" in oos_merged.columns:
        risky = oos_merged[oos_merged["SUM of po_qty"] > oos_merged["SUM of actual_qty"]]
        risky = pd.merge(risky, high_value, on="product_id", how="left")
        st.dataframe(risky[["product_id", "product_name", "SUM of po_qty", "SUM of actual_qty", "FR", "quantity_sold"]].dropna())
    else:
        st.warning("OOS sheet missing PO qty columns.")

# ----------------------
# Vendor Scorecard Tab
# ----------------------
with tab4:
    st.header("Vendor Fulfillment Scorecard")
    if "FR" in oos_merged.columns:
        vendor_fr = oos_merged.groupby("product_id").agg({
            "FR": "mean",
            "product_name": "first"
        }).reset_index()
        st.subheader("Average FR by Product")
        st.dataframe(vendor_fr.sort_values("FR", ascending=True))
    else:
        st.warning("FR column missing in OOS data.")

# ----------------------
# Forecast Accuracy Tab
# ----------------------
with tab5:
    st.header("Forecast Accuracy")
    forecast_df = gv.copy()
    if "Forecast Qty" in forecast_df.columns and "Actual Sales (Qty)" in forecast_df.columns:
        forecast_df["Error"] = (forecast_df["Forecast Qty"] - forecast_df["Actual Sales (Qty)"]).abs()
        forecast_df["APE"] = forecast_df["Error"] / (forecast_df["Actual Sales (Qty)"] + 1e-9)
        forecast_df["MAPE"] = forecast_df.groupby("product_id")["APE"].transform("mean") * 100
        top_forecast = forecast_df[["product_id", "product_name", "MAPE"]].drop_duplicates().sort_values("MAPE")
        st.dataframe(top_forecast.head(15))
    else:
        st.warning("Forecast or actual sales data not available in the dataset.")

    csv = acc_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Accuracy Summary CSV", csv, file_name="forecast_accuracy_summary.csv", mime="text/csv")


