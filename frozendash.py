import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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

# Rename the date column in FR to match oos
vendor.rename(columns={"request_shipping_date: Day": "Date"}, inplace=True)

# Ensure 'Date' column is in datetime format
vendor["Date"] = pd.to_datetime(vendor["Date"], errors="coerce")
oos["Date"] = pd.to_datetime(oos["Date"], errors="coerce")

# Group FR by product_id and Date to get daily FR

fr_grouped = vendor.groupby(["product_id", "Date"]).agg({
    "SUM of po_qty": "sum",
    "SUM of actual_qty": "sum"
}).reset_index()

# Force numeric conversion
fr_grouped["SUM of actual_qty"] = pd.to_numeric(fr_grouped["SUM of actual_qty"], errors="coerce")
fr_grouped["SUM of po_qty"] = pd.to_numeric(fr_grouped["SUM of po_qty"], errors="coerce")

# Now safely calculate FR
fr_grouped["FR"] = fr_grouped["SUM of actual_qty"] / fr_grouped["SUM of po_qty"]

# Calculate FR
fr_grouped["FR"] = fr_grouped["SUM of actual_qty"] / fr_grouped["SUM of po_qty"]
fr_grouped.replace([np.inf, -np.inf], np.nan, inplace=True)

# Merge into OOS
oos_merged = pd.merge(oos, fr_grouped[["product_id", "Date", "FR"]], on=["product_id", "Date"], how="left")

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
oos_merged = pd.merge(
    oos,
    fr_grouped[["product_id", "Date", "SUM of po_qty", "SUM of actual_qty", "FR"]],
    on=["product_id", "Date"],
    how="left"
)

# Optional: fill FR NaNs with 0 if needed
oos_merged["FR"] = oos_merged["FR"].fillna(0)
# Merge GV + OOS for Forecast
merged_gv_oos = pd.merge(
    gv,
    oos,
    left_on=["product_id", "date_key"],
    right_on=["product_id", "Date"],
    how="left"
)

# Then bring in FR from fr_grouped
merged_gv_oos = pd.merge(
    merged_gv_oos,
    fr_grouped[["product_id", "Date", "FR"]],
    on=["product_id", "Date"],
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
# ----------------------
# GV Trend Tab (Tab 2)
# ----------------------
with tab2:
    st.header("Goods Value (GV) Trend")
    st.write(gv.columns.tolist())

    # Filter to Last 30 Days
    last_date = gv["date_key"].max()
    cutoff_date = gv["date_key"].max() - pd.Timedelta(days=30)
    gv_l30 = gv[gv["date_key"] >= cutoff_date]
    
    # Group by date and l1_category
    category_daily = gv_l30.groupby(["date_key", "l1_category_name"])["goods_value"].sum().reset_index()
    
    # Plot
    fig = px.line(
        category_daily,
        x="date_key", y="goods_value",
        color="l1_category_name",
        title="GV Trend by L1 Category (Last 30 Days)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# OOS Risk Tab
# ----------------------
# ----------------------
# OOS Risk Tab
# ----------------------
with tab3:
    st.header("Out-of-Stock Risk Items")

    # Choose lookback period
    days_back = st.slider("Lookback Period (Days)", 7, 90, 30)
    threshold = st.slider("Min Qty Threshold", 0, 50, 10)

    latest_date = oos_merged["Date"].max()
    cutoff_date = latest_date - pd.Timedelta(days=days_back)
    oos_recent = oos_merged[oos_merged["Date"] >= cutoff_date]

    if {"SUM of po_qty", "SUM of actual_qty"}.issubset(oos_recent.columns):
        risky = oos_recent[oos_recent["SUM of po_qty"] > oos_recent["SUM of actual_qty"]]
        risky = pd.merge(risky, daily_agg[["product_id", "quantity_sold"]], on="product_id", how="left")
        risky = risky[risky["quantity_sold"] >= threshold]

        st.subheader("🔎 Risky SKUs")
        st.dataframe(risky[[
            "product_id", "product_name", "Date", "SUM of po_qty", "SUM of actual_qty", "FR", "quantity_sold"
        ]].dropna())

        # 1️⃣ Top 10 worst FR products
        st.subheader("📌 Top 10 Worst Fulfillment SKUs")
        worst_fulfillment = risky.groupby("product_id").agg({
            "FR": "mean",
            "product_name": "first",
            "SUM of po_qty": "sum",
            "SUM of actual_qty": "sum"
        }).reset_index()
        worst_fulfillment = worst_fulfillment.sort_values("FR").head(10)
        st.dataframe(worst_fulfillment)

        st.subheader("🚨 Operationally Risky SKUs (Based on Stock & Forecast)")

        # Prepare daily L30 data
        daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
        cutoff = daily["Date"].max() - pd.Timedelta(days=days_back)
        daily_l30 = daily[daily["Date"] >= cutoff].copy()
    
        # Ensure numeric
        for col in ["DOI Hub", "Stock WH", "Stock HUB", "Actual Sales (Qty)", "Forecast Qty"]:
            daily_l30[col] = pd.to_numeric(daily_l30[col], errors="coerce")
    
        # Define risky logic: low DOI + not enough HUB stock vs forecast + low WH stock
        risky_operational = daily_l30[
            (daily_l30["DOI Hub"] < 2) &
            (daily_l30["Forecast Qty"] > daily_l30["Stock HUB"]) &
            (daily_l30["Stock WH"] < threshold)
        ]
    
        if not risky_operational.empty:
            st.dataframe(
                risky_operational[[
                    "Date", "Product Name", "Vendor Name", "DOI Hub", "Stock WH", "Stock HUB", "Forecast Qty"
                ]].sort_values("Forecast Qty", ascending=False).dropna().head(20)
            )
        else:
            st.info("No operationally risky SKUs found in the selected period.")
    
        # 2️⃣ Vendor-level FR summary
        if "Vendor Name" in oos_recent.columns:
                st.subheader("🏭 Vendor-Level FR (Last N Days)")
                vendor_fr = risky.groupby("Vendor Name").agg({
                    "FR": "mean",
                    "product_id": "nunique"
                }).reset_index().rename(columns={"product_id": "Total Risky SKUs"})
                st.dataframe(vendor_fr.sort_values("FR"))
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
        forecast_df = forecast_df.dropna(subset=["APE"])

        mape_df = forecast_df.groupby("product_id").agg({
            "APE": "mean",
            "product_name_x": "first",  # Use correct column name
            "FR": "mean"
        }).reset_index()
        
        # Rename for clean output
        mape_df = mape_df.rename(columns={"product_name_x": "product_name"})
        mape_df["MAPE"] = mape_df["APE"] * 100

        acc_df = mape_df[["product_id", "product_name", "MAPE", "FR"]].sort_values("MAPE")

        st.dataframe(acc_df.head(15))
        csv = acc_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Forecast Accuracy CSV", csv, file_name="forecast_accuracy_summary.csv", mime="text/csv")
    else:
        st.warning("Forecast or actual sales data not available in the dataset.")

