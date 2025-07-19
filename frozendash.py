import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

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
    "oos": f"{base_url}&gid=1511488791",
}

@st.cache_data(ttl=600)
def load_csv(url, header=0):
    return pd.read_csv(url, header=header)

# Load data
df_gv = load_csv(urls["gv"])[['L1', 'product_id', 'Product Name', 'PARETO', 'Mar', 'May', 'Jun', 'Jul']]
df_vendor = load_csv(urls["vendor"], header=1)
df_oos = load_csv(urls["oos"], header=0)
df_daily = pd.concat([
    load_csv(urls["daily_3t"], header=1),
    load_csv(urls["daily_seafood"], header=1),
    load_csv(urls["daily_daging"], header=1),
    load_csv(urls["daily_ayam"], header=1)
])

# Clean and prepare daily sales
july_cols = [col for col in df_daily.columns if re.match(r"^\d+ Jul$", col)]
df_daily["Total_July_Sales"] = df_daily[july_cols].sum(axis=1, numeric_only=True)

# Merge GV and daily
merged = df_gv.merge(df_daily[["SKU Numbers", "Total_July_Sales"]], left_on="product_id", right_on="SKU Numbers", how="left")
merged = merged.merge(df_vendor[['L1', 'Vendor Name', 'FR']], on='L1', how='left')

# Clean vendor fill rate (FR)
df_vendor["FR"] = df_vendor["FR"].astype(str).str.replace("%", "", regex=False).astype(float) / 100

# DOI calculation
merged['DOI'] = merged['Jul'] / merged['Total_July_Sales'].replace(0, np.nan)

# GV Trend
st.header("📈 GV Trend Analysis")
fig, ax = plt.subplots()
for _, row in merged.iterrows():
    x = [3, 5, 6, 7]
    y = [row['Mar'], row['May'], row['Jun'], row['Jul']]
    ax.plot(x, y, alpha=0.3)
plt.xticks([3, 5, 6, 7], ['Mar', 'May', 'Jun', 'Jul'])
plt.title("GV Trend per SKU")
st.pyplot(fig)

# High potential SKUs (positive slope)
st.subheader("High Potential SKUs")
def compute_slope(row):
    x = np.array([3, 5, 6, 7]).reshape(-1, 1)
    y = np.array([row['Mar'], row['May'], row['Jun'], row['Jul']])
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

merged["slope"] = merged.apply(compute_slope, axis=1)
high_potential = merged[merged["slope"] > 0].sort_values("slope", ascending=False).head(10)
st.dataframe(high_potential[["product_id", "Product Name", "slope", "Jul"]])

# Vendor Scorecard
st.header("🏆 Vendor Scorecard")
vendor_score = merged.groupby("Vendor Name")["FR"].mean().dropna().sort_values(ascending=False).head(10)
st.bar_chart(vendor_score)

# Correlation matrix for substitution
st.header("🔄 SKU Substitution Matrix")
correlation_matrix = df_daily[july_cols].T.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# DOI vs FR scatter
st.header("📊 DOI vs Fill Rate")
fig, ax = plt.subplots()
ax.scatter(merged['DOI'], merged['FR'])
ax.set_xlabel("DOI")
ax.set_ylabel("Fill Rate")
ax.set_title("DOI vs Fill Rate")
st.pyplot(fig)

# Summary: High value but OOS risk SKUs
st.header("⚠️ High Value, OOS Risk SKUs")
df_oos['%OOS Today'] = df_oos['%OOS Today'].astype(str).str.replace("%", "", regex=False).astype(float)
high_oos_risk = df_oos[(df_oos['%OOS Today'] > 50) & (df_oos['Stock WH'] <= 0)]
high_oos_risk = high_oos_risk.merge(merged[['product_id', 'Jul']], left_on='Product ID', right_on='product_id', how='left')
high_oos_risk = high_oos_risk.sort_values('Jul', ascending=False).head(10)
st.dataframe(high_oos_risk[["Product ID", "Product Name", "%OOS Today", "Jul"]])

# Summary: Worst performing vendors
st.header("📉 Worst Performing Vendors")
worst_vendors = df_vendor.sort_values("FR").head(10)
st.dataframe(worst_vendors[["Vendor Name", "FR"]])



