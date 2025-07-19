import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Frozen SKU Dashboard", layout="wide")

# --- URLs by gid ---
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

# --- Load data ---
df_gv = load_csv(urls["gv"])[['L1', 'product_id', 'Product Name', 'PARETO', 'Mar', 'May', 'Jun', 'Jul']]
df_vendor = load_csv(urls["vendor"], header=1)
df_oos = load_csv(urls["oos"])
df_daily = pd.concat([
    load_csv(urls["daily_3t"], header=1),
    load_csv(urls["daily_seafood"], header=1),
    load_csv(urls["daily_daging"], header=1),
    load_csv(urls["daily_ayam"], header=1),
], ignore_index=True)

# --- Clean & Convert ---
df_gv["product_id"] = df_gv["product_id"].astype(str)
df_daily["SKU Numbers"] = df_daily["SKU Numbers"].astype(str)
df_oos["product_id"] = df_oos["Product ID"].astype(str)

df_oos["Stock WH"] = pd.to_numeric(df_oos["Stock WH"], errors="coerce")
df_oos["DOI Hub"] = pd.to_numeric(df_oos["DOI Hub"], errors="coerce")

df_vendor["FR"] = (
    df_vendor["FR"].astype(str).str.replace("%", "", regex=False).str.strip()
)
df_vendor["FR"] = pd.to_numeric(df_vendor["FR"], errors="coerce") / 100

# --- Total July Sales ---
july_cols = [col for col in df_daily.columns if re.match(r"^\d{1,2} Jul$", col)]
df_daily["Total_July_Sales"] = df_daily[july_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

# --- Sales Slope ---
df_daily["Sales Slope"] = np.nan
x = np.arange(len(july_cols)).reshape(-1, 1)

for i, row in df_daily.iterrows():
    y = pd.to_numeric(row[july_cols], errors="coerce").values
    if np.isnan(y).all() or np.count_nonzero(~np.isnan(y)) < 2:
        continue
    model = LinearRegression().fit(x[~np.isnan(y)], y[~np.isnan(y)])
    df_daily.at[i, "Sales Slope"] = model.coef_[0]

# --- Merge ---
merged = df_gv.merge(df_daily[["SKU Numbers", "Total_July_Sales", "Sales Slope"]],
                     left_on="product_id", right_on="SKU Numbers", how="left")
merged = merged.merge(df_oos[['product_id', 'Stock WH']], on='product_id', how='left')
merged["DOI"] = merged["Stock WH"] / merged["Total_July_Sales"] * 7
merged["DOI"] = merged["DOI"].replace([np.inf, -np.inf], np.nan)

# --- Vendor Filter ---
vendor_list = df_vendor["Vendor Name"].dropna().unique().tolist()
selected_vendor = st.sidebar.selectbox("Select Vendor", ["All"] + vendor_list)

if selected_vendor != "All":
    merged = merged[merged["L1"] == selected_vendor]

# --- Tabs ---
summary, trends, vendor_tab, correlation_tab = st.tabs(["Summary", "GV Trend", "Vendor Scorecard", "Correlation"])

# --- Summary Tab ---
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

# --- GV Trend Tab ---
with trends:
    st.subheader("GV Trend & High Potential SKUs")
    try:
        st.line_chart(merged[["Mar", "May", "Jun", "Jul"]].mean(numeric_only=True))
    except Exception as e:
        st.warning(f"Trend chart error: {e}")

    st.markdown("**High Potential SKUs (Positive Sales Slope)**")
    if "Sales Slope" in merged.columns:
        slope_df = merged[merged["Sales Slope"] > 0].copy()
        slope_df = slope_df[["Product Name", "Sales Slope"]].dropna().sort_values("Sales Slope", ascending=False)
        st.dataframe(slope_df)
    else:
        st.warning("No slope data available.")

# --- Vendor Scorecard Tab ---
with vendor_tab:
    st.subheader("Vendor Fill Rate Scorecard")
    if "FR" in df_vendor.columns:
        st.bar_chart(df_vendor.set_index("Vendor Name")["FR"].dropna())
    else:
        st.warning("FR data missing.")

# --- Correlation Tab ---
with correlation_tab:
    st.subheader("Correlation Matrix")
    corr_df = merged[["Mar", "May", "Jun", "Jul", "Stock WH", "Total_July_Sales", "DOI"]].apply(pd.to_numeric, errors="coerce").dropna()
    if not corr_df.empty:
        corr = corr_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Insufficient data for correlation matrix.")

    st.subheader("DOI vs Fill Rate")
    doi_vs_fr = merged.merge(df_vendor[["Vendor Name", "FR"]], left_on="L1", right_on="Vendor Name", how="left")
    doi_vs_fr = doi_vs_fr.dropna(subset=["DOI", "FR"])
    if not doi_vs_fr.empty:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=doi_vs_fr, x="DOI", y="FR", hue="PARETO", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("Not enough data for DOI vs FR scatter plot.")

st.sidebar.write("GV rows:", df_gv.shape[0])
st.sidebar.write("Daily rows:", df_daily.shape[0])
st.sidebar.write("Daily July columns:", july_cols)
st.sidebar.write("Total_July_Sales not null:", df_daily["Total_July_Sales"].notna().sum())
st.sidebar.write("Sales Slope not null:", df_daily["Sales Slope"].notna().sum())
st.sidebar.write("Merged rows after vendor filter:", merged.shape[0])
st.sidebar.write("Merged DOI not null:", merged["DOI"].notna().sum())
