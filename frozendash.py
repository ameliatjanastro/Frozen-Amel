import streamlit as st
import pandas as pd
import numpy as np

# Load data
df_gv = pd.read_excel("GV.xlsx", sheet_name="GV", usecols="A:J")
df_vendor = pd.read_excel("GV.xlsx", sheet_name="vendor")
df_daily = pd.read_excel("GV.xlsx", sheet_name="daily")

# Clean column names
df_gv.columns = df_gv.columns.str.strip()
df_vendor.columns = df_vendor.columns.str.strip()
df_daily.columns = df_daily.columns.str.strip()

# Filter out missing product_id
df_gv = df_gv[df_gv['product_id'].notna()]

# Filter: only July columns (e.g., '1 Jul', '2 Jul', ...)
july_cols = [col for col in df_daily.columns if 'Jul' in col]
df_daily['Total July Sales'] = df_daily[july_cols].sum(axis=1)

# Calculate DOI
df_daily['DOI'] = df_daily[july_cols].apply(lambda row: (row > 0).sum(), axis=1)

# Merge GV with Daily
df = pd.merge(df_gv, df_daily[['SKU Numbers', 'Total July Sales', 'DOI']], 
              left_on='product_id', right_on='SKU Numbers', how='left')

# Merge with Vendor
df = pd.merge(df, df_vendor[['L1', 'FR']], on='L1', how='left')

# Clean up missing values
df['Total July Sales'] = df['Total July Sales'].fillna(0)
df['DOI'] = df['DOI'].fillna(0)
df['FR'] = df['FR'].fillna(0)

# Calculate GV Slope
month_cols = ['Mar', 'May', 'Jun', 'Jul']
df['GV_Slope'] = df[month_cols].apply(lambda row: np.polyfit(range(len(month_cols)), row, 1)[0] if row.notna().sum() > 1 else 0, axis=1)

# Sidebar filters
category_options = df['L1'].dropna().unique()
pareto_options = ['All'] + sorted([p for p in df['PARETO'].dropna().unique() if p != ''])

selected_category = st.sidebar.selectbox("Category", sorted(category_options))
selected_pareto = st.sidebar.selectbox("Pareto", pareto_options)

# Apply filters
filtered_df = df[df['L1'] == selected_category]
if selected_pareto != 'All':
    filtered_df = filtered_df[filtered_df['PARETO'] == selected_pareto]

# Metrics
total_skus = len(filtered_df)
high_gv_low_doi = filtered_df[(filtered_df['Jul'] > 0) & (filtered_df['DOI'] < 3)]
high_growth_skus = filtered_df[filtered_df['GV_Slope'] > 0]

# Dashboard UI
st.markdown("## 🧊 Frozen SKU Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs", total_skus)
col2.metric("High GV & Low DOI", len(high_gv_low_doi))
col3.metric("High Growth SKUs", len(high_growth_skus))

# 🔥 High GV SKUs with OOS Risk
st.markdown("### 🔥 High GV SKUs with OOS Risk")
st.dataframe(high_gv_low_doi[['L1', 'product_id', 'Product Name', 'Jul', 'DOI', 'GV_Slope']])

# 🌟 High Potential SKUs via positive slope
st.markdown("### 🌟 High Potential SKUs")
st.dataframe(high_growth_skus[['L1', 'product_id', 'Product Name', 'GV_Slope', 'Jul']])

# 🏅 Vendor Scorecard integrated (via FR)
st.markdown("### 🏅 Vendor Scorecard")
vendor_scorecard = filtered_df.groupby('L1')[['FR', 'Total July Sales']].mean().reset_index()
st.dataframe(vendor_scorecard)

# Optional debug
# st.write(df.head())

