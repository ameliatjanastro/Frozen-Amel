import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Frozen Dashboard", layout="wide")

# Define base URL and GIDs
base_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/export?format=csv"

urls = {
    "gv": f"{base_url}&gid=828450040",
    "vendor": f"{base_url}&gid=356668619",
    "daily_3t": f"{base_url}&gid=1396953592",
    "daily_seafood": f"{base_url}&gid=1898295439",
    "daily_daging": f"{base_url}&gid=138270218",
    "daily_ayam": f"{base_url}&gid=1702050586",
}

# Load data
@st.cache_data(ttl=600)
def load_csv(url, header=0):
    return pd.read_csv(url, header=header)

# GV starts at row 1
df_gv = load_csv(urls["gv"])

# Vendor starts from row 2 (i.e., header=1)
df_vendor = load_csv(urls["vendor"], header=1)

# Daily data (multiple sheets), also start from row 2
df_daily = pd.concat([
    load_csv(urls["daily_3t"], header=1),
    load_csv(urls["daily_seafood"], header=1),
    load_csv(urls["daily_daging"], header=1),
    load_csv(urls["daily_ayam"], header=1)
], ignore_index=True)

# Clean GV
month_cols = ['Mar', 'May', 'Jun', 'Jul']
df_gv = df_gv[df_gv['product_id'].notna()]
df_gv[month_cols] = df_gv[month_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

df_gv['GV_Slope'] = df_gv[month_cols].apply(
    lambda row: np.polyfit(range(len(row)), row.values, 1)[0] if (row > 0).sum() >= 2 else 0,
    axis=1
)

# Clean Vendor
df_vendor = df_vendor[df_vendor['L1'].notna()]
df_vendor = df_vendor[['L1', 'Vendor Name', 'FR']]

# Merge GV and Vendor
df = pd.merge(df_gv, df_vendor, on='L1', how='left')

# Clean Daily
df_daily = df_daily[df_daily['SKU Numbers'].notna()]
df_daily['SKU Numbers'] = df_daily['SKU Numbers'].astype(str)
july_cols = [col for col in df_daily.columns if 'July' in col]
df_daily['Total July Sales'] = df_daily[july_cols].fillna(0).sum(axis=1)

# Merge with Daily by Product ID
df = pd.merge(df, df_daily[['SKU Numbers', 'Total July Sales']], left_on='product_id', right_on='SKU Numbers', how='left')

# Fill NaN
df = df.fillna(0)

# Calculate DOH
df['DOH'] = df.apply(lambda x: x['Qty'] / (x['Total July Sales'] / 30) if x['Total July Sales'] > 0 else 0, axis=1)

# Flagging
df['Issue Flag'] = df.apply(lambda x: '⚠️ High GV, Low DOH' if x['Jul'] > 500000 and x['DOH'] < 2 else '', axis=1)

# SIDEBAR FILTERS
st.sidebar.title("Frozen Filter")
vendor_options = ['All'] + sorted(df['Vendor Name'].unique())
selected_vendor = st.sidebar.selectbox("Vendor", vendor_options)

if selected_vendor != 'All':
    filtered_df = df[df['Vendor Name'] == selected_vendor]
else:
    filtered_df = df

# DISPLAY
st.title("🧊 Frozen SKU Dashboard")

st.markdown("### Summary")
st.write(filtered_df[['L1', 'product_id', 'Product Name', 'Jul', 'GV_Slope', 'DOH', 'Issue Flag']].sort_values(by='Jul', ascending=False))

# Metrics
total_sku = len(filtered_df)
high_gv_low_doh = filtered_df[(filtered_df['Jul'] > 500000) & (filtered_df['DOH'] < 2)]
problem_skus = len(high_gv_low_doh)

col1, col2 = st.columns(2)
col1.metric("Total SKUs", total_sku)
col2.metric("High GV & Low DOH", problem_skus)

# Optional: Expand full table
with st.expander("📋 See Full Data Table"):
    st.dataframe(filtered_df)

