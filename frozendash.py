import re
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
    "oos": f"{base_url}&gid=1511488791",
}

@st.cache_data(ttl=600)
def load_csv(url, header=0):
    return pd.read_csv(url, header=header)

# Load datasets
df_gv = load_csv(urls["gv"])[['L1', 'product_id', 'Product Name', 'PARETO', 'Mar', 'May', 'Jun', 'Jul']]
df_vendor = load_csv(urls["vendor"], header=1)
df_oos = load_csv(urls["oos"])
df_daily = pd.concat([
    load_csv(urls["daily_3t"], header=1),
    load_csv(urls["daily_seafood"], header=1),
    load_csv(urls["daily_daging"], header=1),
    load_csv(urls["daily_ayam"], header=1)
], ignore_index=True)

# Fix and clean column names
df_daily.columns = df_daily.columns.str.strip()
df_oos.columns = df_oos.columns.str.strip()
df_vendor.columns = df_vendor.columns.str.strip()
df_gv.columns = df_gv.columns.str.strip()

# Clean July sales
daily_cols = [col for col in df_daily.columns if re.match(r"\d{1,2} Jul", col)]
df_daily[daily_cols] = df_daily[daily_cols].apply(pd.to_numeric, errors='coerce')
df_daily["Total_July_Sales"] = df_daily[daily_cols].sum(axis=1)

# Merge Daily Sales to GV
if "SKU Numbers" in df_daily.columns:
    merged = df_gv.merge(df_daily[["SKU Numbers", "Total_July_Sales"]],
                         left_on="product_id", right_on="SKU Numbers", how="left")
else:
    st.error("SKU Numbers column not found in daily sales data.")
    merged = df_gv.copy()
    merged["Total_July_Sales"] = np.nan

# Merge Fill Rate from Vendor Sheet
if "FR" in df_vendor.columns:
    df_vendor["FR"] = df_vendor["FR"].astype(str).str.replace("%", "").astype(float) / 100
    merged = merged.merge(df_vendor[["L1", "FR", "Vendor Name"]], on="L1", how="left")
else:
    st.warning("FR column not found in vendor sheet.")
    merged["FR"] = np.nan

# Merge stock from OOS sheet
if "Product ID" in df_oos.columns and "Stock WH" in df_oos.columns:
    df_oos["Product ID"] = df_oos["Product ID"].astype(str).str.strip()
    df_oos["Stock WH"] = pd.to_numeric(df_oos["Stock WH"], errors='coerce')
    merged = merged.merge(df_oos[["Product ID", "Stock WH"]],
                          left_on="product_id", right_on="Product ID", how="left")
    merged["current_stock"] = merged["Stock WH"]
else:
    st.warning("Product ID or Stock WH not found in OOS sheet.")
    merged["current_stock"] = np.nan

# Calculate DOI
merged["DOI"] = merged["current_stock"] / merged["Total_July_Sales"].replace(0, np.nan)
merged["DOI"] = merged["DOI"].replace([np.inf, -np.inf], np.nan)

# Simulated GV slope (replace with real trend logic later)
merged["GV_Slope"] = np.random.randn(len(merged))

# ------------------ Streamlit Display ------------------

st.title("Frozen SKU Dashboard")
st.subheader("Data Preview")
st.dataframe(merged.head(30))

# Chart 1: GV Trend (Top 10 by slope)
st.subheader("📈 GV Trend by SKU")
top_slope = merged.sort_values("GV_Slope", ascending=False).head(10)
fig, ax = plt.subplots()
sns.barplot(data=top_slope, y="Product Name", x="GV_Slope", ax=ax)
ax.set_title("Top 10 SKU by GV Trend (Simulated Slope)")
st.pyplot(fig)

# Chart 2: Vendor Fill Rate
st.subheader("🏢 Vendor Fill Rate")
if "Vendor Name" in merged.columns:
    vendor_score = merged.groupby("Vendor Name")["FR"].mean().dropna().sort_values(ascending=False).head(10)
    st.bar_chart(vendor_score)
else:
    st.warning("Vendor Name column not found for vendor scorecard.")

# Chart 3: DOI vs Fill Rate
st.subheader("📉 DOI vs Fill Rate")
scatter_data = merged[["DOI", "FR"]].dropna()
st.scatter_chart(scatter_data)

# Chart 4: Correlation Matrix
st.subheader("📊 Correlation Between Metrics")
numeric_cols = merged.select_dtypes(include=[np.number])
if not numeric_cols.empty:
    corr = numeric_cols.corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
else:
    st.write("No numeric data available for correlation matrix.")

# Notes
st.markdown("---")
st.caption("All metrics are updated based on the latest July sales data and stock WH availability.")

