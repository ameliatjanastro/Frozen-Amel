import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

st.set_page_config(layout="wide")

# === Google Sheets GIDs ===
base_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/export?format=csv"
urls = {
    "gv": f"{base_url}&gid=0",
    "oos": f"{base_url}&gid=1511488791",
    "fr": f"{base_url}&gid=1390643918",
}

@st.cache_data
def load_data():
    gv = pd.read_csv(urls["gv"])
    oos = pd.read_csv(urls["oos"])
    fr = pd.read_csv(urls["fr"])

    gv["date_key"] = pd.to_datetime(gv["date_key"], errors="coerce")
    gv["product_id"] = gv["product_id"].astype(str)

    oos["Date"] = pd.to_datetime(oos["Date"], errors="coerce")
    oos["product_id"] = oos["product_id"].astype(str)
    oos["Forecast Qty"] = pd.to_numeric(oos["Forecast Qty"], errors="coerce")
    oos["Actual Sales (Qty)"] = pd.to_numeric(oos["Actual Sales (Qty)"], errors="coerce")

    fr["product_id"] = fr["product_id"].astype(str)
    fr["FR"] = pd.to_numeric(fr["FR"], errors="coerce")

    # Add product_name from GV if missing
    if oos["product_name"].isnull().any():
        oos = pd.merge(
            oos.drop(columns=["product_name"]),
            gv[["product_id", "product_name"]].drop_duplicates(),
            on="product_id", how="left"
        )

    # Merge FR into OOS
    fr_latest = fr.sort_values("request_shipping_date: Day").drop_duplicates("product_id", keep="last")
    oos = pd.merge(oos, fr_latest[["product_id", "FR"]], on="product_id", how="left")

    return gv, oos, fr

gv, oos, fr = load_data()

# Filter L30 data
max_date = oos["Date"].max()
min_date = max_date - timedelta(days=30)
oos_l30 = oos[(oos["Date"] >= min_date) & (oos["Date"] <= max_date)]

# ==== STREAMLIT MULTI-TAB ====
tab_summary, tab_trend, tab_fr, tab_accuracy = st.tabs(["📋 Summary", "📈 GV Trend", "🏭 Vendor Scorecard", "🎯 Forecast Accuracy"])

with tab_summary:
    st.header("📋 Summary")

    col1, col2 = st.columns(2)
    col1.metric("Total July Sales (Qty)", int(gv["quantity_sold"].sum()))

    valid = oos_l30.dropna(subset=["Forecast Qty", "Actual Sales (Qty)"])
    valid = valid[valid["Forecast Qty"] > 0]
    valid["accuracy"] = 1 - abs(valid["Forecast Qty"] - valid["Actual Sales (Qty)"]) / valid["Forecast Qty"]
    forecast_accuracy = valid["accuracy"].mean()
    col2.metric("📈 Forecast Accuracy (L30)", f"{forecast_accuracy:.2%}")

    st.subheader("⚠️ High GV + OOS Risk SKUs (L30)")
    risky = oos_l30[oos_l30["%OOS Today"] >= 70]
    gv_sum = gv.groupby("product_id")[["goods_value"]].sum().reset_index()
    risky = pd.merge(risky, gv_sum, on="product_id", how="left").sort_values("goods_value", ascending=False)
    st.dataframe(risky[["product_id", "product_name", "Vendor Name", "%OOS Today", "goods_value"]].drop_duplicates("product_id").head(10), use_container_width=True)

with tab_trend:
    st.header("📈 GV Trend")

    vendor_options = gv["l1_category_name"].dropna().unique().tolist()
    product_options = gv["product_name"].dropna().unique().tolist()

    selected_vendor = st.selectbox("Filter by L1 Category (Vendor Tag)", ["All"] + vendor_options)
    selected_product = st.selectbox("Filter by Product Name", ["All"] + product_options)

    filtered_gv = gv.copy()
    if selected_vendor != "All":
        filtered_gv = filtered_gv[filtered_gv["l1_category_name"] == selected_vendor]
    if selected_product != "All":
        filtered_gv = filtered_gv[filtered_gv["product_name"] == selected_product]

    trend = filtered_gv.groupby("date_key")["goods_value"].sum().reset_index()
    st.line_chart(trend.set_index("date_key"))

with tab_fr:
    st.header("🏭 Vendor Scorecard (FR)")
    vendor_score = oos.groupby("Vendor Name")["FR"].mean().reset_index().sort_values("FR")
    st.dataframe(vendor_score, use_container_width=True)

with tab_accuracy:
    st.header("🎯 Forecast Accuracy (L30)")

    # SKU-level accuracy
    acc_df = valid.groupby("product_id").agg({
        "Forecast Qty": "sum",
        "Actual Sales (Qty)": "sum",
        "FR": "mean"
    }).reset_index()
    acc_df["Forecast Deviation"] = acc_df["Forecast Qty"] - acc_df["Actual Sales (Qty)"]
    acc_df["Accuracy"] = 1 - abs(acc_df["Forecast Deviation"]) / acc_df["Forecast Qty"]
    acc_df = pd.merge(acc_df, gv[["product_id", "product_name"]].drop_duplicates(), on="product_id", how="left")
    acc_df = acc_df[["product_id", "product_name", "Forecast Qty", "Actual Sales (Qty)", "Forecast Deviation", "FR", "Accuracy"]]
    acc_df = acc_df.sort_values("Accuracy")

    st.dataframe(acc_df.head(30), use_container_width=True)

    # Export CSV
    csv = acc_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Accuracy Summary CSV", csv, file_name="forecast_accuracy_summary.csv", mime="text/csv")


