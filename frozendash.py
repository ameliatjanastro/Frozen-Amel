import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
import requests

# ---- Google Sheet CSV links
base_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/export?format=csv"
urls = {
    "gv": f"{base_url}&gid=0",
    "vendor": f"{base_url}&gid=1841763572",
    "oos": f"{base_url}&gid=1511488791",
    "fr": f"{base_url}&gid=1570890844"
}

@st.cache_data(ttl=600)
def read_csv_url(url):
    return pd.read_csv(url)

# ---- Load Data
gv = read_csv_url(urls["gv"])
vendor = read_csv_url(urls["vendor"])
oos = read_csv_url(urls["oos"])
fr = read_csv_url(urls["fr"])

# ---- Clean Dates
gv['date_key'] = pd.to_datetime(gv['date_key'], errors='coerce')
oos['Date'] = pd.to_datetime(oos['Date'], errors='coerce')
fr['request_shipping_date: Day'] = pd.to_datetime(fr['request_shipping_date: Day'], errors='coerce')

# ---- L30 Filter
cutoff_date = pd.Timestamp.today() - pd.Timedelta(days=30)
gv = gv[gv['date_key'] >= cutoff_date]
oos = oos[oos['Date'] >= cutoff_date]
fr = fr[fr['request_shipping_date: Day'] >= cutoff_date]

# ---- Merge FR with OOS
fr_summary = fr.groupby('product_id').agg({
    'SUM of po_qty': 'sum',
    'SUM of actual_qty': 'sum'
}).reset_index()
fr_summary['FR'] = (fr_summary['SUM of actual_qty'] / fr_summary['SUM of po_qty']).replace([np.inf, -np.inf], 0)

oos = oos.merge(fr_summary, on='product_id', how='left')

# ---- Add product_name from GV
if 'product_name' not in oos.columns or oos['product_name'].isnull().all():
    oos = oos.merge(gv[['product_id', 'product_name']].drop_duplicates(), on='product_id', how='left')

# ---- Forecast Deviation
oos['Forecast Dev'] = oos['Forecast Qty'] - oos['Actual Sales (Qty)']

# ---- Sidebar Filters
with st.sidebar:
    selected_vendor = st.selectbox("Select Vendor", ["All"] + sorted(oos['Vendor Name'].dropna().unique().tolist()))
    selected_product = st.selectbox("Select Product", ["All"] + sorted(oos['product_name'].dropna().unique().tolist()))

filtered_gv = gv.copy()
if selected_vendor != "All":
    filtered_gv = filtered_gv.merge(oos[['product_id', 'Vendor Name']], on='product_id', how='left')
    filtered_gv = filtered_gv[filtered_gv['Vendor Name'] == selected_vendor]

if selected_product != "All":
    filtered_gv = filtered_gv[filtered_gv['product_name'] == selected_product]

# ---- Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Trend", "🏪 Vendor Scorecard", "📦 Forecast Accuracy"])

with tab1:
    st.header("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total July Sales", int(filtered_gv["quantity_sold"].sum()))
    col2.metric("Total Goods Value", f"Rp {filtered_gv['goods_value'].sum():,.0f}")
    col3.metric("Unique SKUs", filtered_gv['product_id'].nunique())

    # High-value but OOS-risk SKUs
    high_value_oos = oos[oos['%OOS Today'] > 0]
    high_value_oos = high_value_oos.merge(gv[['product_id', 'goods_value']], on='product_id', how='left')
    top_risks = high_value_oos.groupby(['product_id', 'product_name'])['goods_value'].sum().reset_index()
    top_risks = top_risks.sort_values(by='goods_value', ascending=False).head(10)
    
    st.subheader("High-Value but OOS-Risk SKUs (L30)")
    st.dataframe(top_risks)

with tab2:
    st.header("Goods Value Trend (Last 30 Days)")
    trend = filtered_gv.groupby('date_key')['goods_value'].sum().reset_index()
    st.line_chart(trend.set_index('date_key'))

with tab3:
    st.header("Vendor Scorecard")
    vendor['FR'] = pd.to_numeric(vendor['FR'], errors='coerce')
    vendor_sorted = vendor.sort_values('FR')
    st.dataframe(vendor_sorted[['Vendor Name', 'Total SKU', 'SUM of Request Qty', 'SUM of Actual Qty', 'FR']])

with tab4:
    st.header("Forecast Accuracy")
    fr_display = oos[['product_id', 'product_name', 'Vendor Name', 'Forecast Qty', 'Actual Sales (Qty)', 'Forecast Dev', 'FR']]
    st.dataframe(fr_display.sort_values(by='FR', ascending=True))

    # Export accuracy
    csv = fr_display.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Accuracy Summary", data=csv, file_name='forecast_accuracy.csv', mime='text/csv')

    # Export CSV
    csv = acc_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Full Accuracy Summary CSV", csv, file_name="forecast_accuracy_summary.csv", mime="text/csv")


