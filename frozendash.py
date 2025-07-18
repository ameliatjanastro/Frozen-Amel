import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Sheet configuration
# -----------------------------
sheet_url = "https://docs.google.com/spreadsheets/d/1P9ntTYxuCOmTeBgG4UKD0fnRUp1Ixne5AeSycHg0Gnw/edit#gid=0"
sheet_id = sheet_url.split("/d/")[1].split("/")[0]

sheet_gv = "GV"
sheet_vendor = "vendor"
sheet_daily = "daily"
sheet_oos = "OOS Daily"

# -----------------------------
# Load data
# -----------------------------
@st.cache_data(ttl=600)
def load_data():
    url_gv = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_gv}"
    url_vendor = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_vendor}"
    url_daily = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_daily}"
    url_oos = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_oos}"
    
    return (
        pd.read_csv(url_gv),
        pd.read_csv(url_vendor),
        pd.read_csv(url_daily),
        pd.read_csv(url_oos)
    )

df_gv, df_vendor, df_daily, df_oos = load_data()

# -----------------------------
# Process Daily Sheet
# -----------------------------
july_cols = [col for col in df_daily.columns if re.search(r"\bJul\b", col)]
df_daily[july_cols] = df_daily[july_cols].apply(pd.to_numeric, errors='coerce')
df_daily["Total_July_Sales"] = df_daily[july_cols].sum(axis=1)

# -----------------------------
# Merge GV + Daily + Vendor
# -----------------------------
df = df_gv.merge(df_daily[["SKU Numbers", "Total_July_Sales"]], left_on="product_id", right_on="SKU Numbers", how="left")
df = df.merge(df_vendor[["L1", "FR"]], on="L1", how="left")
df["FR"] = df["FR"].str.replace("%", "", regex=False).astype(float) / 100

# -----------------------------
# Merge Stock WH (from OOS Daily)
# -----------------------------
df_oos.columns = df_oos.columns.str.strip()
if "Product ID" in df_oos.columns and "Stock WH" in df_oos.columns:
    df = df.merge(df_oos[["Product ID", "Stock WH"]], left_on="product_id", right_on="Product ID", how="left")
    df["current_stock"] = df["Stock WH"]
else:
    df["current_stock"] = np.nan

# -----------------------------
# Calculate DOI
# -----------------------------
df["DOI"] = df["current_stock"] / df["Total_July_Sales"].replace(0, np.nan)
df["DOI"] = df["DOI"].replace([np.inf, -np.inf], np.nan)

# -----------------------------
# Streamlit Dashboard
# -----------------------------
st.title("🧊 Frozen SKU Dashboard")

# Preview main data
st.subheader("📦 Preview")
st.write(df[["product_id", "Product Name", "Total_July_Sales", "current_stock", "DOI", "FR"]].head(10))

# -----------------------------
# 📈 1. GV Trend Insight
# -----------------------------
st.subheader("📈 GV Trend Insight (Positive Slope SKUs)")
df["GV_Slope"] = pd.to_numeric(df["GV_Slope"], errors="coerce")
top_gv = df.sort_values("GV_Slope", ascending=False).head(10)
st.write(top_gv[["product_id", "Product Name", "GV_Slope", "Total_July_Sales"]])

# -----------------------------
# 📊 2. Vendor Scorecard (FR Summary)
# -----------------------------
st.subheader("📊 Vendor Scorecard")
vendor_summary = df.groupby("Vendor Name").agg({
    "product_id": "count",
    "Total_July_Sales": "sum",
    "FR": "mean"
}).rename(columns={"product_id": "Total SKU", "FR": "Avg FR"})
vendor_summary["Avg FR"] = vendor_summary["Avg FR"].round(2)
st.dataframe(vendor_summary.sort_values("Avg FR", ascending=False))

# -----------------------------
# 🔁 3. Substitution Potential (Correlation)
# -----------------------------
st.subheader("🔁 SKU Substitution Correlation Matrix")
sales_matrix = df_daily.set_index("SKU Numbers")[july_cols].dropna(how="any")
corr_matrix = sales_matrix.T.corr()

fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# -----------------------------
# 📉 4. FR vs DOI Chart
# -----------------------------
st.subheader("📉 DOI vs FR Scatter Plot")
plot_df = df[["FR", "DOI"]].dropna()
st.scatter_chart(plot_df)

# -----------------------------
# 📌 Optional Filters
# -----------------------------
with st.expander("🔍 Filter by Vendor"):
    vendor_filter = st.multiselect("Select Vendor(s)", df["Vendor Name"].dropna().unique())
    if vendor_filter:
        filtered_df = df[df["Vendor Name"].isin(vendor_filter)]
        st.write(filtered_df[["product_id", "Product Name", "Total_July_Sales", "DOI", "FR"]])
