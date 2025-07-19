import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Frozen SKU Dashboard", layout="wide")

# --- Google Sheets URLs ---
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

# --- Load and clean data ---
df_gv = load_csv(urls["gv"])[['L1', 'product_id', 'Product Name', 'PARETO', 'Mar', 'May', 'Jun', 'Jul']]
df_vendor = load_csv(urls["vendor"], header=1)
df_oos = load_csv(urls["oos"])
df_daily = pd.concat([
    load_csv(urls["daily_3t"], header=1),
    load_csv(urls["daily_seafood"], header=1),
    load_csv(urls["daily_daging"], header=1),
    load_csv(urls["daily_ayam"], header=1),
], ignore_index=True)

# --- Safe conversions ---
df_vendor["FR"] = pd.to_numeric(
    df_vendor["FR"].astype(str).str.replace("%", "").str.strip(), errors="coerce"
) / 100
df_gv["Jul"] = pd.to_numeric(df_gv["Jul"], errors="coerce")
df_oos["Stock WH"] = pd.to_numeric(df_oos.get("Stock WH", np.nan), errors="coerce")

# --- Total July sales ---
daily_july_cols = [col for col in df_daily.columns if 'Jul' in col]
df_daily["Total July Sales"] = df_daily[daily_july_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

# --- Merge datasets ---
merged = df_gv.merge(df_daily[["SKU Numbers", "Total July Sales"]], left_on="product_id", right_on="SKU Numbers", how="left")
merged = merged.merge(df_oos[["Product Name", "Stock WH"]], on="Product Name", how="left")

# --- Sales slope ---
def calculate_slope(row):
    y = row[daily_july_cols].values.astype(float)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

merged[daily_july_cols] = df_daily[daily_july_cols].apply(pd.to_numeric, errors="coerce")
merged["Sales Slope"] = df_daily[daily_july_cols].apply(calculate_slope, axis=1)

# --- DOI Calculation ---
merged["DOI"] = merged["Stock WH"] / merged["Total July Sales"] * 31

# --- Dashboard layout ---
tabs = st.tabs(["Summary", "GV Trend", "Vendor Scorecard", "Correlation"])

with tabs[0]:
    st.header("🔎 Summary")

    # High-value but OOS risk SKUs
    high_risk = merged[(merged["Jul"] > 0) & (merged["Stock WH"] <= 0)]
    st.subheader("High Value but OOS Risk SKUs")
    st.dataframe(high_risk.sort_values("Jul", ascending=False)[["Product Name", "Jul", "Stock WH"]])

    # Worst vendors by FR
    st.subheader("Worst Performing Vendors by FR")
    st.dataframe(df_vendor.sort_values("FR").head(10)[["Vendor Name", "FR"]])

with tabs[1]:
    st.header("📈 GV Trend")
    st.line_chart(df_gv.set_index("Product Name")[['Mar', 'May', 'Jun', 'Jul']].apply(pd.to_numeric, errors="coerce"))

    st.subheader("High-Potential SKUs (Positive Sales Slope)")
    if "Sales Slope" in merged.columns:
        st.dataframe(
            merged[merged["Sales Slope"] > 0][["Product Name", "Sales Slope"]]
            .sort_values("Sales Slope", ascending=False)
        )

with tabs[2]:
    st.header("📊 Vendor Scorecard")
    fig, ax = plt.subplots()
    sns.barplot(
        data=df_vendor.sort_values("FR", ascending=False),
        x="FR",
        y="Vendor Name",
        palette="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

with tabs[3]:
    st.header("📌 Correlation Matrix")
    numeric_cols = merged[["Jul", "Total July Sales", "Stock WH", "DOI"]].apply(pd.to_numeric, errors="coerce")
    corr = numeric_cols.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("DOI vs FR Scatter")
    scatter_df = merged.merge(df_vendor, left_on="L1", right_on="L1", how="left")
    scatter_df = scatter_df[["DOI", "FR"]].dropna()
    fig2, ax2 = plt.subplots()
    ax2.scatter(scatter_df["DOI"], scatter_df["FR"])
    ax2.set_xlabel("DOI")
    ax2.set_ylabel("FR")
    st.pyplot(fig2)

st.success("Dashboard loaded successfully ✅")

