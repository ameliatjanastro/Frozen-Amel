import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load Google Sheets data
sheet_url_base = "https://docs.google.com/spreadsheets/d/"
sheet_id = "1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw"
gv_gid = "1289742955"
vendor_gid = "330721172"
daily_gid = "1487401490"
oos_gid = "1511488791"

gv_url = f"{sheet_url_base}{sheet_id}/export?format=csv&gid={gv_gid}"
vendor_url = f"{sheet_url_base}{sheet_id}/export?format=csv&gid={vendor_gid}"
daily_url = f"{sheet_url_base}{sheet_id}/export?format=csv&gid={daily_gid}"
oos_url = f"{sheet_url_base}{sheet_id}/export?format=csv&gid={oos_gid}"

df_gv = pd.read_csv(gv_url)
df_vendor = pd.read_csv(vendor_url)
df_daily = pd.read_csv(daily_url)
df_oos = pd.read_csv(oos_url)

# Clean FR percentage
df_vendor["FR"] = df_vendor["FR"].astype(str).str.replace("%", "").astype(float) / 100

# Compute Total July Sales from '1 Jul' to '7 Jul' columns
july_days = [f"{i} Jul" for i in range(1, 8)]
df_daily["Total_July_Sales"] = df_daily[july_days].sum(axis=1)

# Merge data sources
df_gv["product_id"] = df_gv["product_id"].astype(str)
df_daily["SKU Numbers"] = df_daily["SKU Numbers"].astype(str)
merged = df_gv.merge(df_daily[["SKU Numbers", "Total_July_Sales"]],
                     left_on="product_id", right_on="SKU Numbers", how="left")

df_oos["Product ID"] = df_oos["Product ID"].astype(str)
merged = merged.merge(df_oos[["Product ID", "Stock WH", "current_stock"]],
                      left_on="product_id", right_on="Product ID", how="left")

# Calculate DOI (Days of Inventory)
merged["DOI"] = merged["Stock WH"] / merged["Total_July_Sales"]
merged["DOI"] = merged["DOI"].replace([np.inf, -np.inf], np.nan)

# Calculate GV trend (slope of GV across months)
month_cols = ['Mar', 'May', 'Jun', 'Jul']
def compute_slope(row):
    y = row[month_cols].values
    x = np.arange(len(month_cols)).reshape(-1, 1)
    if np.isnan(y).any():
        return np.nan
    model = LinearRegression().fit(x, y)
    return model.coef_[0]
merged["GV_Slope"] = merged.apply(compute_slope, axis=1)

# Filter for vendor selection
st.title("Frozen SKU Dashboard")
vendor_options = merged["Vendor Name"].dropna().unique()
selected_vendor = st.selectbox("Select a Vendor", options=vendor_options)
filtered = merged[merged["Vendor Name"] == selected_vendor]

# Display metrics
st.metric("Total July Sales", round(filtered["Total_July_Sales"].sum(), 2))
st.metric("Average DOI", round(filtered["DOI"].mean(), 2))
st.metric("Top SKU by GV Growth", filtered.sort_values("GV_Slope", ascending=False).iloc[0]["Product Name"])

# Charts
st.subheader("GV Trend by SKU")
for _, row in filtered.iterrows():
    plt.plot(month_cols, row[month_cols], label=row["Product Name"])
plt.xlabel("Month")
plt.ylabel("GV")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(plt)

st.subheader("Top Vendors by Fill Rate (FR)")
vendor_score = df_vendor.groupby("Vendor Name")["FR"].mean().dropna().sort_values(ascending=False).head(10)
st.bar_chart(vendor_score)

# Correlation between SKUs
st.subheader("Correlation Matrix - Daily Sales")
sales_data = df_daily[july_days].dropna()
corr = sales_data.T.corr()
fig, ax = plt.subplots()
sns.heatmap(corr, ax=ax, cmap="coolwarm")
st.pyplot(fig)

# DOI vs FR Scatter
st.subheader("DOI vs Fill Rate")
doi_fr = merged.merge(df_vendor[["Vendor Name", "FR"]], on="Vendor Name", how="left")
fig, ax = plt.subplots()
sns.scatterplot(data=doi_fr, x="DOI", y="FR", hue="Vendor Name", legend=False, ax=ax)
st.pyplot(fig)

