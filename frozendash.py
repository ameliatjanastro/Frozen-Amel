import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Frozen SKU Dashboard", layout="wide")
st.title("🧊 Frozen SKU Dashboard")

# URLs (CSV export links from GIDs)
base_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/export?format=csv"
urls = {
    "gv": f"{base_url}&gid=0",
    "vendor": f"{base_url}&gid=1841763572",
    "oos": f"{base_url}&gid=1511488791"
}

@st.cache_data(ttl=600)
def load_csv(url, header=0):
    return pd.read_csv(url, header=header)

# Load data
df_gv = load_csv(urls["gv"])
df_vendor = load_csv(urls["vendor"], header=1)
df_oos = load_csv(urls["oos"])

# Clean column names and formats
df_gv.columns = df_gv.columns.str.strip()
df_vendor.columns = df_vendor.columns.str.strip()
df_oos.columns = df_oos.columns.str.strip()

# Identify daily columns for July in OOS Daily
daily_cols = [col for col in df_oos.columns if "Jul" in col]
df_oos[daily_cols] = df_oos[daily_cols].apply(pd.to_numeric, errors='coerce')

# --- Insight 1: GV Trend + Total July Sales + Sales Slope ---
july_sales = df_oos[['SKU Numbers', 'Product Name'] + daily_cols].copy()
july_sales['Total_July_Sales'] = july_sales[daily_cols].sum(axis=1)

# Linear regression to get sales slope
july_sales = july_sales.dropna(subset=['Total_July_Sales'])
X = np.arange(len(daily_cols)).reshape(-1, 1)
slope_list = []

for _, row in july_sales.iterrows():
    y = row[daily_cols].values.astype(float)
    if np.isnan(y).all():
        slope = np.nan
    else:
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
    slope_list.append(slope)

july_sales['Sales Slope'] = slope_list

# Merge GV with July data
merged = pd.merge(df_gv, july_sales[['SKU Numbers', 'Total_July_Sales', 'Sales Slope']],
                  left_on='product_id', right_on='SKU Numbers', how='left')

# --- Insight 2: Vendor Scorecard via FR ---
df_vendor['FR'] = pd.to_numeric(df_vendor['FR'].astype(str).str.replace('%',''), errors='coerce') / 100

# --- Insight 3: DOI Calculation ---
# Simplified DOI = Stock WH / Avg daily sales July
stock_wh = df_oos[['SKU Numbers', 'Stock WH']].drop_duplicates()
stock_wh['Stock WH'] = pd.to_numeric(stock_wh['Stock WH'], errors='coerce')

merged = pd.merge(merged, stock_wh, on='SKU Numbers', how='left')
merged['Avg_Daily_Sales'] = merged['Total_July_Sales'] / len(daily_cols)
merged['DOI'] = merged['Stock WH'] / merged['Avg_Daily_Sales']

# --- Insight 4: High-potential SKUs ---
st.subheader("🔥 High-Potential SKUs (Positive Sales Slope)")
high_potential = merged[merged['Sales Slope'] > 0].sort_values(by='Sales Slope', ascending=False)
st.dataframe(high_potential[['Product Name', 'Sales Slope', 'Total_July_Sales', 'DOI']].dropna())

# --- Insight 5: Vendor Scorecard ---
st.subheader("📦 Vendor Scorecard")
st.dataframe(df_vendor[['Vendor Name', 'Total SKU', 'SUM of Request Qty', 'SUM of Actual Qty', 'FR']].dropna())

# --- Insight 6: Worst Performing Vendors (low FR) ---
st.subheader("🚨 Worst Performing Vendors")
worst_vendors = df_vendor[df_vendor['FR'] < 0.8].sort_values('FR')
st.dataframe(worst_vendors[['Vendor Name', 'FR']])

# --- Insight 7: OOS Risk Summary ---
st.subheader("❄️ High Value but OOS Risk SKUs")
oos_risk = merged[(merged['Total_July_Sales'] > 100) & (merged['DOI'] < 2)]
st.dataframe(oos_risk[['Product Name', 'Total_July_Sales', 'DOI']].dropna())

# --- Charts ---
st.subheader("📈 GV Trend by Month")
month_cols = ['Mar', 'May', 'Jun', 'Jul']
for col in month_cols:
    merged[col] = pd.to_numeric(merged[col], errors='coerce')

trend_avg = merged[month_cols].mean().reset_index()
trend_avg.columns = ['Month', 'Average GV']
sns.set_style("whitegrid")
fig, ax = plt.subplots()
sns.lineplot(data=trend_avg, x='Month', y='Average GV', marker='o', ax=ax)
ax.set_title("Average GV Over Time")
st.pyplot(fig)

# --- Insight 8: DOI vs FR ---
st.subheader("🔍 DOI vs Vendor FR Scatter Plot")
merged_vendor = pd.merge(merged, df_vendor[['L1', 'FR']], on='L1', how='left')
fig2, ax2 = plt.subplots()
sns.scatterplot(data=merged_vendor, x='DOI', y='FR', hue='PARETO', ax=ax2)
ax2.set_title("DOI vs FR")
st.pyplot(fig2)

