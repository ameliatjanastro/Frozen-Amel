import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Frozen SKU Dashboard", layout="wide")

# URLs to Google Sheets (CSV export links)
base_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/export?format=csv"
urls = {
    "gv": f"{base_url}&gid=828450040",
    "vendor": f"{base_url}&gid=356668619",
    "daily_3t": f"{base_url}&gid=1396953592",
    "daily_seafood": f"{base_url}&gid=1898295439",
    "daily_daging": f"{base_url}&gid=138270218",
    "daily_ayam": f"{base_url}&gid=1702050586",
}

@st.cache_data(ttl=600)
def load_csv(url, header=0):
    return pd.read_csv(url, header=header)

# Load data
df_gv = load_csv(urls["gv"])[['L1', 'product_id', 'Product Name', 'PARETO', 'Mar', 'May', 'Jun', 'Jul']]
st.write(df["GV"].describe())
df_vendor = load_csv(urls["vendor"], header=1)
df_daily = pd.concat([
    load_csv(urls["daily_3t"], header=1),
    load_csv(urls["daily_seafood"], header=1),
    load_csv(urls["daily_daging"], header=1),
    load_csv(urls["daily_ayam"], header=1)
], ignore_index=True)

# Clean GV
df_gv = df_gv[df_gv['product_id'].notna()]
month_cols = ['Mar', 'May', 'Jun', 'Jul']
df_gv[month_cols] = df_gv[month_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
df_gv['GV_Slope'] = df_gv[month_cols].apply(
    lambda row: np.polyfit(range(len(row)), row.values, 1)[0] if (row > 0).sum() >= 2 else 0, axis=1)

# Clean vendor
df_vendor = df_vendor[df_vendor['L1'].notna()][['L1', 'Vendor Name', 'FR']]

# Merge GV + Vendor
df = pd.merge(df_gv, df_vendor, on='L1', how='left')

df["Jul"] = pd.to_numeric(df["Jul"], errors='coerce')

st.subheader("🔍 Debug: Raw Data Preview")
st.write(df.head(10))  # or st.dataframe(df)

# Daily
df['PARETO'] = df['PARETO'].fillna('Unknown')
df_daily = df_daily[df_daily['SKU Numbers'].notna()]
df_daily['SKU Numbers'] = df_daily['SKU Numbers'].astype(str)
df['product_id'] = df['product_id'].astype(str)
july_cols = [col for col in df_daily.columns if 'July' in col]
df_daily['Total July Sales'] = df_daily[july_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

# Merge with daily sales
df = pd.merge(df, df_daily[['SKU Numbers', 'Total July Sales']], left_on='product_id', right_on='SKU Numbers', how='left')
df = df.fillna(0)

# Add DOH (formerly labeled DOH but is DOI)
df['DOI'] = df.apply(lambda x: x['Jul'] / (x['Total July Sales'] / 30) if x['Total July Sales'] > 0 else 0, axis=1)
df["DOI"] = pd.to_numeric(df["DOI"], errors='coerce')

# Flag risky SKUs
df['Issue Flag'] = df.apply(lambda x: '🔥' if x['Jul'] > 500000 and x['DOI'] < 2 else '', axis=1)

# Sidebar Filters
st.sidebar.title("Filters")
category_options = ['All'] + sorted(df['L1'].unique())
pareto_options = ['All'] + sorted(df['PARETO'].unique())
selected_cat = st.sidebar.selectbox("Category", category_options)
selected_pareto = st.sidebar.selectbox("Pareto", pareto_options)

filtered_df = df.copy()
if selected_cat != 'All':
    filtered_df = filtered_df[filtered_df['L1'] == selected_cat]
if selected_pareto != 'All':
    filtered_df = filtered_df[filtered_df['PARETO'] == selected_pareto]


# After filters
st.subheader("🔍 Debug: Filtered Data")
st.write(filtered_df.head(10))

# Optional: show unique PARETO or Category values
st.write("Unique Pareto:", df['PARETO'].unique())
st.write("Unique Category:", df['Category'].unique())

# Main Dashboard
st.title("🧊 Frozen SKU Dashboard")

# Key Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs", len(filtered_df))
col2.metric("High GV & Low DOI", len(filtered_df[filtered_df['Issue Flag'] == '🔥']))
col3.metric("High Growth SKUs", len(filtered_df[filtered_df['GV_Slope'] > 0]))

# High GV Issues
st.subheader("🔥 High GV SKUs with OOS Risk")
st.write(filtered_df[filtered_df['Issue Flag'] == '🔥'][['L1', 'product_id', 'Product Name', 'Jul', 'DOI', 'GV_Slope']])

# Growth Slope
st.subheader("🌟 High Potential SKUs")
st.write(filtered_df[filtered_df['GV_Slope'] > 0].sort_values(by='GV_Slope', ascending=False)[['L1', 'product_id', 'Product Name', 'GV_Slope']].head(10))

# Vendor Scorecard
st.subheader("🏅 Vendor Scorecard")
# Ensure 'FR' is numeric
df['FR'] = pd.to_numeric(df['FR'], errors='coerce')

# Replace missing vendor names to avoid groupby NaN key
df['Vendor Name'] = df['Vendor Name'].fillna('Unknown')

# Now it's safe to group and average
vendor_scorecard = df.groupby('Vendor Name')[['FR']].mean().sort_values(by='FR', ascending=False)
vendor_scorecard = df.groupby('Vendor Name')[['FR']].mean().sort_values(by='FR', ascending=False)
st.dataframe(vendor_scorecard)

# Substitute Detection (Correlation)
st.subheader("🔁 Substitute Detection (Correlation Preview)")
sales_matrix = df.pivot_table(index='product_id', values='Total July Sales', columns='L1', aggfunc='sum').fillna(0)
correlation_matrix = sales_matrix.corr(method='pearson')
st.write("Pairwise Category Correlation (proxy for substitution potential):")
st.dataframe(correlation_matrix)

# Category Trend
st.subheader("📈 Daily GV Trend by Category")
category_sums = df.groupby('L1')[['Mar', 'May', 'Jun', 'Jul']].sum()
fig, ax = plt.subplots(figsize=(12, 5))
for cat in category_sums.index:
    ax.plot(['Mar', 'May', 'Jun', 'Jul'], category_sums.loc[cat], label=cat)
ax.set_ylabel("GV (IDR)")
ax.set_title("GV Trend by Category")
ax.legend()
st.pyplot(fig)

# Full Table
with st.expander("📋 Full Table"):
    st.dataframe(filtered_df.sort_values(by='Jul', ascending=False))

