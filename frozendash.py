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
], ignore_index=True)

# Clean merge keys
df_gv["product_id"] = df_gv["product_id"].astype(str).str.strip()
df_daily["SKU Numbers"] = df_daily["SKU Numbers"].astype(str).str.strip()
df_oos["SKU Numbers"] = df_oos["SKU Numbers"].astype(str).str.strip()

# Show matching product IDs count for debug
common_ids = set(df_gv["product_id"]).intersection(set(df_daily["SKU Numbers"]))
st.sidebar.write("Common product IDs between GV and daily:", len(common_ids))

# Filter July columns
july_cols = [col for col in df_daily.columns if re.match(r"\d{1,2} Jul", col)]

# Convert July daily sales to float
for col in july_cols:
    df_daily[col] = pd.to_numeric(df_daily[col], errors='coerce')

# Calculate Total July Sales
df_daily["Total_July_Sales"] = df_daily[july_cols].sum(axis=1)

# Calculate sales slope
def calculate_slope(row):
    y = row[july_cols].values
    x = np.arange(1, len(july_cols) + 1)
    if np.all(np.isnan(y)) or np.count_nonzero(~np.isnan(y)) < 2:
        return np.nan
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    return model.coef_[0]

df_daily["Sales Slope"] = df_daily.apply(calculate_slope, axis=1)

# Merge GV with daily
merged = df_gv.merge(
    df_daily[["SKU Numbers", "Total_July_Sales", "Sales Slope"] + july_cols],
    left_on="product_id",
    right_on="SKU Numbers",
    how="left"
)

# Merge with OOS to get Stock WH
df_oos["Stock WH"] = pd.to_numeric(df_oos["Stock WH"], errors='coerce')
merged = merged.merge(df_oos[["SKU Numbers", "Stock WH"]], on="SKU Numbers", how="left")

# Calculate DOI
merged["DOI"] = merged["Stock WH"] / (merged["Total_July_Sales"] / 31)
merged["DOI"] = merged["DOI"].replace([np.inf, -np.inf], np.nan)

# Clean FR column
df_vendor["FR"] = df_vendor["FR"].astype(str).str.replace('%', '').str.strip()
df_vendor["FR"] = pd.to_numeric(df_vendor["FR"], errors='coerce') / 100

# Sidebar filter
vendor_filter = st.sidebar.selectbox("Filter by Vendor", options=["All"] + list(df_vendor["Vendor Name"].dropna().unique()))
if vendor_filter != "All":
    filtered_vendor = df_vendor[df_vendor["Vendor Name"] == vendor_filter]
    merged = merged[merged["L1"].isin(filtered_vendor["L1"])]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Summary", "GV Trend", "Vendor Scorecard", "Correlation"])

with tab1:
    st.subheader("High Potential SKUs (Positive Sales Slope)")
    high_slope = merged[merged["Sales Slope"] > 0][["Product Name", "Sales Slope"]].sort_values("Sales Slope", ascending=False)
    st.dataframe(high_slope, use_container_width=True)

    st.subheader("OOS Risk SKUs with High GV")
    oos_risk = merged[(merged["DOI"] < 1) & (merged["Jul"] > 100)]
    st.dataframe(oos_risk[["Product Name", "DOI", "Jul", "Stock WH"]], use_container_width=True)

with tab2:
    st.subheader("GV Trend")
    trend_df = df_gv.melt(id_vars=["Product Name"], value_vars=["Mar", "May", "Jun", "Jul"],
                          var_name="Month", value_name="GV")
    trend_df["GV"] = pd.to_numeric(trend_df["GV"], errors='coerce')
    avg_trend = trend_df.groupby("Month")["GV"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.lineplot(data=avg_trend, x="Month", y="GV", marker="o", ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Vendor Scorecard (FR)")
    vendor_score = df_vendor[["Vendor Name", "FR"]].dropna().sort_values("FR", ascending=False)
    st.dataframe(vendor_score, use_container_width=True)

with tab4:
    st.subheader("Correlation Matrix: Daily Sales")
    corr_data = df_daily[july_cols].dropna(how='any')
    if not corr_data.empty:
        corr = corr_data.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough data for correlation matrix.")

    st.subheader("DOI vs FR")
    merged_with_fr = merged.merge(df_vendor[["L1", "FR"]], on="L1", how="left")
    fig, ax = plt.subplots()
    temp_df = merged_with_fr.dropna(subset=["DOI", "FR"])
    sns.scatterplot(data=temp_df, x="FR", y="DOI", ax=ax)
    st.pyplot(fig)
