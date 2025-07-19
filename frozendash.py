import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Frozen SKU Dashboard", layout="wide")

# Google Sheet CSV URLs (via gid)
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
df_oos = load_csv(urls["oos"])
df_daily = pd.concat([
    load_csv(urls["daily_3t"], header=1),
    load_csv(urls["daily_seafood"], header=1),
    load_csv(urls["daily_daging"], header=1),
    load_csv(urls["daily_ayam"], header=1),
], ignore_index=True)

# Clean and convert types

df_gv["product_id"] = df_gv["product_id"].astype(str)
df_daily["SKU Numbers"] = df_daily["SKU Numbers"].astype(str)
df_oos["product_id"] = df_oos["product_id"].astype(str)
df_oos["Stock WH"] = pd.to_numeric(df_oos["Stock WH"], errors="coerce")
df_oos["DOI Hub"] = pd.to_numeric(df_oos["DOI Hub"], errors="coerce")

# Clean vendor FR %
df_vendor["FR"] = (
    df_vendor["FR"].astype(str)
    .str.replace("%", "", regex=False)
    .replace("nan", np.nan)
    .astype(float) / 100
)

# Calculate Total July Sales
july_cols = [col for col in df_daily.columns if re.match(r"^\d{1,2} Jul$", col)]
df_daily["Total_July_Sales"] = df_daily[july_cols].sum(axis=1, min_count=1)

# Merge GV with Daily Sales
merged = df_gv.merge(df_daily[["SKU Numbers", "Total_July_Sales"]],
                     left_on="product_id", right_on="SKU Numbers", how="left")

# Calculate DOI (Days of Inventory)
merged = merged.merge(df_oos[['product_id', 'Stock WH']], on='product_id', how='left')
merged["DOI"] = merged["Stock WH"] / merged["Total_July_Sales"] * 7

# Score trend using linear regression
merged["Sales Slope"] = 0.0
x = np.arange(len(july_cols)).reshape(-1, 1)

for i, row in df_daily.iterrows():
    y = row[july_cols].values.astype(float)
    if np.isnan(y).all():
        continue
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        continue
    model = LinearRegression().fit(x[mask], y[mask])
    df_daily.at[i, "Sales Slope"] = model.coef_[0]

# Merge slope
merged = merged.merge(df_daily[["SKU Numbers", "Sales Slope"]],
                      left_on="product_id", right_on="SKU Numbers", how="left")

# Filter by vendor
vendor_list = df_vendor["Vendor Name"].unique().tolist()
selected_vendor = st.sidebar.selectbox("Select Vendor", ["All"] + vendor_list)

if selected_vendor != "All":
    merged = merged[merged["L1"] == selected_vendor]

# Tabbed layout
summary, trends, vendor_tab, correlation_tab = st.tabs(["Summary", "GV Trend", "Vendor Scorecard", "Correlation"])

with summary:
    st.subheader("Summary Table")
    high_value_oos = merged[(merged["DOI"] < 1) & (merged["Jul"] > 100)]
    worst_vendors = df_vendor.sort_values("FR").head(5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**High-Value but OOS-Risk SKUs**")
        st.dataframe(high_value_oos[["Product Name", "Jul", "Stock WH", "DOI"]])
    with col2:
        st.markdown("**Worst Performing Vendors (by FR)**")
        st.dataframe(worst_vendors[["Vendor Name", "FR"]])

with trends:
    st.subheader("GV Trend & High Potential SKUs")
    st.line_chart(merged[["Mar", "May", "Jun", "Jul"]].mean())
    st.markdown("**High Potential SKUs (Positive Sales Slope)**")
    st.dataframe(merged[merged["Sales Slope"] > 0][["Product Name", "Sales Slope"]].sort_values("Sales Slope", ascending=False))

with vendor_tab:
    st.subheader("Vendor Fill Rate Scorecard")
    st.bar_chart(df_vendor.set_index("Vendor Name")["FR"])

with correlation_tab:
    st.subheader("Correlation Matrix")
    numeric = merged[["Mar", "May", "Jun", "Jul", "Stock WH", "Total_July_Sales", "DOI"]].dropna()
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("DOI vs Fill Rate")
    doi_vs_fr = merged.merge(df_vendor[["Vendor Name", "FR"]], left_on="L1", right_on="Vendor Name", how="left")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=doi_vs_fr, x="DOI", y="FR", hue="PARETO", ax=ax2)
    st.pyplot(fig2)


