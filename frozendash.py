import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# === SETUP ===
st.set_page_config(page_title="Frozen SKU Dashboard", layout="wide")
st.title("❄️ Frozen SKU Performance Dashboard")

# === Helper: Load Google Sheet via GID ===
def load_sheet_by_gid(gsheet_id, gid):
    url = f"https://docs.google.com/spreadsheets/d/{gsheet_id}/export?format=csv&gid={gid}"
    return pd.read_csv(url)

# === Load Data ===
SHEET_ID = "1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw"
GID_MAP = {
    "GV": "959531066",
    "Vendor": "939076218",
    "Daily": "1394741442",
    "OOS": "1511488791"
}

df_gv = load_sheet_by_gid(SHEET_ID, GID_MAP["GV"])
df_vendor = load_sheet_by_gid(SHEET_ID, GID_MAP["Vendor"])
df_daily = load_sheet_by_gid(SHEET_ID, GID_MAP["Daily"])
df_oos = load_sheet_by_gid(SHEET_ID, GID_MAP["OOS"])

# === PREPROCESSING ===
# Clean column names
df_gv.columns = df_gv.columns.str.strip()
df_vendor.columns = df_vendor.columns.str.strip()
df_daily.columns = df_daily.columns.str.strip()
df_oos.columns = df_oos.columns.str.strip()

# Clean FR
df_vendor["FR"] = df_vendor["FR"].astype(str).str.replace("%", "").astype(float) / 100

# Total July Sales
july_cols = [col for col in df_daily.columns if "Jul" in col and col[0].isdigit()]
df_daily["Total_July_Sales"] = df_daily[july_cols].sum(axis=1)

# Merge GV and daily
df_merge = df_gv.merge(
    df_daily[["SKU Numbers", "Total_July_Sales"]],
    left_on="product_id", right_on="SKU Numbers", how="left"
)

# Merge with Vendor
df_merge = df_merge.merge(
    df_vendor[["L1", "Vendor Name", "FR"]],
    on="L1", how="left"
)

# Merge OOS stock WH
if "Product ID" in df_oos.columns:
    df_oos = df_oos.rename(columns={"Product ID": "product_id"})
df_merge = df_merge.merge(df_oos[["product_id", "Stock WH"]], on="product_id", how="left")

# Calculate DOI = Stock WH / Avg July Sales
df_merge["DOI"] = df_merge["Stock WH"] / (df_merge["Total_July_Sales"] / 31)
df_merge["DOI"] = df_merge["DOI"].replace([np.inf, -np.inf], np.nan)

# Filter: Vendor Selector
vendor_selected = st.selectbox("Select a Vendor", ["All"] + sorted(df_merge["Vendor Name"].dropna().unique().tolist()))
if vendor_selected != "All":
    df_filtered = df_merge[df_merge["Vendor Name"] == vendor_selected]
else:
    df_filtered = df_merge.copy()

# === INSIGHTS SECTION ===

# 📈 GV Trend (Sales Slope)
st.subheader("📈 GV Trend - SKU Sales Slope")
slope_results = []
for _, row in df_filtered.iterrows():
    monthly_cols = ['Mar', 'May', 'Jun', 'Jul']
    try:
        y = row[monthly_cols].astype(float).values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        slope = model.coef_[0]
        slope_results.append((row['Product Name'], slope))
    except:
        continue
slope_df = pd.DataFrame(slope_results, columns=["Product Name", "Slope"]).sort_values(by="Slope", ascending=False)
st.dataframe(slope_df.head(10))

# 🧮 Vendor Scorecard
st.subheader("🧾 Vendor Scorecard (Top 10 by FR)")
vendor_score = df_merge.groupby("Vendor Name")["FR"].mean().dropna().sort_values(ascending=False).head(10)
st.dataframe(vendor_score)

# 🔁 Correlation Matrix (Top 10 Selling SKUs)
st.subheader("📊 Correlation Matrix - SKU Substitution Potential")
top_skus = df_daily.sort_values(by="Total_July_Sales", ascending=False).head(10)
sales_data = top_skus[july_cols].T
sales_data.columns = top_skus["Product Name"].values
corr_matrix = sales_data.corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 📉 DOI vs FR Scatter
st.subheader("📉 Days of Inventory vs Fill Rate")
scatter_df = df_filtered[["DOI", "FR"]].dropna()
fig2, ax2 = plt.subplots()
sns.scatterplot(data=scatter_df, x="DOI", y="FR", ax=ax2)
st.pyplot(fig2)

# 🆘 High-Value but OOS Risk SKUs
st.subheader("🚨 High Value but OOS Risk SKUs")
high_value_oos = df_filtered[(df_filtered["DOI"] < 2) & (df_filtered["Total_July_Sales"] > 100)]
st.dataframe(high_value_oos[["Product Name", "Total_July_Sales", "DOI", "Stock WH"]])

# ❌ Worst Performing Vendors
st.subheader("❌ Worst Performing Vendors")
worst_vendors = df_merge.groupby("Vendor Name")["FR"].mean().dropna().sort_values().head(10)
st.dataframe(worst_vendors)



