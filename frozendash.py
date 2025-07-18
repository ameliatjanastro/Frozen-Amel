import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# === CONFIG ===
st.set_page_config(page_title="Frozen Insight Dashboard", layout="wide")
st.title("🧊 Frozen Category Insight Dashboard")

# === LOAD FROM GOOGLE SHEETS ===
sheet_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/export?format=csv&gid=507762004"
df = pd.read_csv(sheet_url)

# === BASIC CLEANING ===
month_cols = [col for col in df.columns if col in ['May', 'Jun', 'Jul']]
daily_cols = [col for col in df.columns if '2025-' in col]

df[month_cols] = df[month_cols].fillna(0)
df[daily_cols] = df[daily_cols].fillna(0)
df['DOH'] = df['DOH'].fillna(0)

# === DERIVED COLUMNS ===
df['GV_Slope'] = df[month_cols].apply(lambda row: np.polyfit(range(len(row)), row, 1)[0], axis=1)
df['Issue Flag'] = df[['DOH']].apply(lambda x: x['DOH'] < 2, axis=1)

# === SIDEBAR FILTERS ===
st.sidebar.title("📁 Filters")
selected_category = st.sidebar.selectbox("Category", df['Category'].dropna().unique())
selected_vendor = st.sidebar.selectbox("Vendor", ["All"] + sorted(df['Vendor'].dropna().unique()))

# === FILTERED DATA ===
filtered_df = df[df['Category'] == selected_category]
if selected_vendor != "All":
    filtered_df = filtered_df[filtered_df['Vendor'] == selected_vendor]

# === INSIGHT CARDS ===
st.markdown("### 📌 Quick Insights")
col1, col2, col3, col4 = st.columns(4)

try:
    top_oos_sku = filtered_df[(filtered_df['Jul'] > 500000) & (filtered_df['DOH'] < 2)].sort_values(by='Jul', ascending=False).iloc[0]
    col1.metric("🔥 Top OOS Risk SKU", top_oos_sku['Product Name'], f"GV: {int(top_oos_sku['Jul']):,}")
except: col1.write("No OOS risk")

try:
    top_growth_sku = filtered_df.sort_values(by='GV_Slope', ascending=False).iloc[0]
    col2.metric("📈 Top Growth SKU", top_growth_sku['Product Name'], f"Slope: {top_growth_sku['GV_Slope']:.2f}")
except: col2.write("No growth data")

try:
    top_vendor = filtered_df.groupby('Vendor')['Jul'].sum().idxmax()
    top_vendor_val = filtered_df.groupby('Vendor')['Jul'].sum().max()
    col3.metric("🏅 Top Vendor", top_vendor, f"GV: {int(top_vendor_val):,}")
except: col3.write("No vendor data")

try:
    substitutes = filtered_df[daily_cols].T.corr().stack().reset_index()
    substitutes.columns = ['SKU_1', 'SKU_2', 'Correlation']
    substitutes = substitutes[substitutes['SKU_1'] != substitutes['SKU_2']]
    subs_pair = substitutes[substitutes['Correlation'] < -0.5].sort_values(by='Correlation').iloc[0]
    col4.metric("🔁 Possible Substitution", f"{subs_pair['SKU_1']} & {subs_pair['SKU_2']}", f"Corr: {subs_pair['Correlation']:.2f}")
except: col4.write("No substitution risk")

# === SECTION 1: HIGH GV SKUs with OOS Risk ===
st.subheader("🔥 High GV SKUs with Low Stock / OOS Risk")
high_gv_oos = filtered_df[(filtered_df['Jul'] > 500000) & (filtered_df['DOH'] < 2)]
st.dataframe(high_gv_oos[['Product Name', 'Vendor', 'Jul', 'DOH', 'Category']])

# === SECTION 2: High Pareto SKUs with Issues ===
st.subheader("🚨 Pareto X / A SKUs with Issues")
pareto_issues = filtered_df[(filtered_df['Pareto'].isin(['X', 'A'])) & (filtered_df['Issue Flag'])]
st.dataframe(pareto_issues[['Product Name', 'Vendor', 'Pareto', 'Jul', 'DOH']])

# === SECTION 3: Daily GV Trend Monitoring ===
st.subheader("📈 Daily GV Trend - Category View")
daily_sum = filtered_df[daily_cols].sum().reset_index()
daily_sum.columns = ['Date', 'GV']
daily_sum['Date'] = pd.to_datetime(daily_sum['Date'])
fig = px.line(daily_sum, x='Date', y='GV', title=f"GV Trend for Category: {selected_category}")
st.plotly_chart(fig, use_container_width=True)

# === SECTION 4: High-Potential SKU Monitor ===
st.subheader("🌟 Monitoring High-Potential SKUs")
high_potential = filtered_df[(filtered_df['GV_Slope'] > 0) & (filtered_df['Jul'] > 500000)]
st.dataframe(high_potential[['Product Name', 'Vendor', 'GV_Slope', 'Jul', 'DOH']])

# === SECTION 5: Vendor Scorecard ===
st.subheader("🏅 Vendor Scorecard")
vendor_summary = filtered_df.groupby('Vendor').agg({
    'Jul': 'sum',
    'GV_Slope': 'mean',
    'DOH': 'mean'
}).reset_index().sort_values(by='Jul', ascending=False)
st.dataframe(vendor_summary)

# === SECTION 6: Potential Substitute Detection ===
st.subheader("🔁 Substitute Risk / Cannibalization")
if len(filtered_df) >= 2:
    correlation_matrix = filtered_df[daily_cols].T.corr()
    np.fill_diagonal(correlation_matrix.values, np.nan)
    pairs = correlation_matrix.stack().reset_index()
    pairs.columns = ['SKU_1', 'SKU_2', 'Correlation']
    substitutes = pairs[pairs['Correlation'] < -0.5].sort_values(by='Correlation')
    if not substitutes.empty:
        st.write("Detected possible substitution behavior:")
        st.dataframe(substitutes.head(10))
    else:
        st.info("No strong negative correlations detected between SKUs.")
else:
    st.warning("Not enough SKUs to compute correlation matrix.")
