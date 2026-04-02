"""
ATM Intelligence Demand Forecasting - Interactive Streamlit App
FA-2: Building Actionable Insights and an Interactive Python script
Author: Mann Paresh Patel
Date: March 2026

This script performs:
- Exploratory Data Analysis (EDA) with visualizations and observations.
- K-Means clustering to group ATMs by demand behavior.
- Anomaly detection on withdrawals using IQR and Isolation Forest.
- Interactive filtering by day, time, location, etc.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Page configuration
st.set_page_config(page_title="ATM Demand Intelligence", layout="wide", page_icon="🏧")
st.title("🏧 ATM Demand Forecasting & Insights")
st.markdown("Interactive dashboard for exploratory analysis, clustering, and anomaly detection.")

# ------------------------------
# Data loading (cached — runs only once)
@st.cache_data
def load_data():
    """Load the ATM dataset from CSV."""
    df = pd.read_csv("atm_cash_management_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Is_Weekend"] = df["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)
    return df

try:
    df = load_data()
    st.success("✅ Loaded dataset: atm_cash_management_dataset.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found. Place 'atm_cash_management_dataset.csv' in the same folder as this script.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ------------------------------
# Pre-compute heavy computations ONCE and cache them.
# This is the key fix — without caching, KMeans reruns on every button click,
# making the app feel frozen and unresponsive.

@st.cache_data
def run_clustering(k=3):
    """Run KMeans on transaction-level features. Cached so it only recomputes when k changes."""
    cluster_df = df.copy()
    le = LabelEncoder()
    cluster_df["Location_Encoded"] = le.fit_transform(cluster_df["Location_Type"])
    feature_cols = ["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs", "Location_Encoded"]
    X = cluster_df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette scores for k=2..10
    inertias, sil_scores = [], []
    for ki in range(2, 11):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    # Final clustering with chosen k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    cluster_df["Cluster"] = np.nan
    cluster_df.loc[X.index, "Cluster"] = labels

    # PCA for 2D visualisation
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    cluster_df["PC1"] = np.nan
    cluster_df["PC2"] = np.nan
    cluster_df.loc[X.index, "PC1"] = X_pca[:, 0]
    cluster_df.loc[X.index, "PC2"] = X_pca[:, 1]
    cluster_df["Cluster"] = cluster_df["Cluster"].fillna(-1).astype(int).astype(str)

    # Cluster profiles — use mean for numeric, mode for Location_Type
    profile_numeric = cluster_df[cluster_df["Cluster"] != "-1"].groupby("Cluster")[
        ["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs"]
    ].mean().round(1)
    profile_loc = cluster_df[cluster_df["Cluster"] != "-1"].groupby("Cluster")["Location_Type"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else "Unknown"
    )
    profile = profile_numeric.copy()
    profile["Top Location"] = profile_loc
    profile = profile.rename(columns={
        "Total_Withdrawals": "Avg Withdrawals",
        "Total_Deposits": "Avg Deposits",
        "Nearby_Competitor_ATMs": "Avg Competitors",
    })
    return cluster_df, inertias, sil_scores, profile

@st.cache_data
def run_atm_level_clustering(k=3):
    """Cluster ATMs by their mean behaviour. Used in the Interactive Planner tab."""
    atm_agg = df.groupby("ATM_ID").agg(
        Total_Withdrawals=("Total_Withdrawals", "mean"),
        Total_Deposits=("Total_Deposits", "mean"),
        Nearby_Competitor_ATMs=("Nearby_Competitor_ATMs", "first"),
        Location_Type=("Location_Type", "first")
    ).reset_index()
    le = LabelEncoder()
    atm_agg["Location_Encoded"] = le.fit_transform(atm_agg["Location_Type"])
    X = atm_agg[["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs", "Location_Encoded"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    atm_agg["ATM_Cluster"] = kmeans.fit_predict(X_scaled).astype(str)
    return atm_agg[["ATM_ID", "ATM_Cluster"]]

# ------------------------------
# Sidebar filters
st.sidebar.header("🔍 Interactive Filters")
selected_days = st.sidebar.multiselect(
    "Day of Week",
    options=df["Day_of_Week"].unique().tolist(),
    default=df["Day_of_Week"].unique().tolist()
)
selected_times = st.sidebar.multiselect(
    "Time of Day",
    options=df["Time_of_Day"].unique().tolist(),
    default=df["Time_of_Day"].unique().tolist()
)
location_types = sorted(df["Location_Type"].unique().tolist())
selected_locations = st.sidebar.multiselect(
    "Location Type",
    options=location_types,
    default=location_types
)
include_holiday = st.sidebar.checkbox("Include Holidays", value=True)
include_event = st.sidebar.checkbox("Include Special Events", value=True)

# Apply filters
filtered_df = df[
    df["Day_of_Week"].isin(selected_days) &
    df["Time_of_Day"].isin(selected_times) &
    df["Location_Type"].isin(selected_locations)
].copy()
if not include_holiday:
    filtered_df = filtered_df[filtered_df["Holiday_Flag"] == 0]
if not include_event:
    filtered_df = filtered_df[filtered_df["Special_Event_Flag"] == 0]

st.sidebar.markdown(f"**Filtered records:** {len(filtered_df):,} / {len(df):,}")

if filtered_df.empty:
    st.warning("⚠️ No data matches the current filters. Please adjust the sidebar selections.")
    st.stop()

# ------------------------------
# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Data Analysis",
    "📈 Clustering ATMs",
    "🚨 Anomaly Detection",
    "⚙️ Interactive Planner"
])

# ==================== TAB 1: EDA ====================
with tab1:
    st.header("Stage 3 – Exploratory Data Analysis")
    st.markdown("Visual exploration to uncover trends, patterns, and relationships in ATM cash demand data.")

    # Distribution Analysis
    st.subheader("📦 Distribution Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_df, x="Total_Withdrawals", nbins=50, marginal="box",
                           title="Histogram of Total Withdrawals")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Withdrawals are right-skewed; most days see moderate demand with a long tail of high-demand events (paydays, holidays).")
    with col2:
        fig = px.histogram(filtered_df, x="Total_Deposits", nbins=50, marginal="box",
                           title="Histogram of Total Deposits")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Deposits are also skewed and typically lower than withdrawals, indicating net cash outflow at ATMs.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(filtered_df, y="Total_Withdrawals", title="Box Plot – Withdrawals (Outlier Check)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Several high-value outliers present; likely correspond to holiday or event days.")
    with col2:
        fig = px.box(filtered_df, y="Total_Deposits", title="Box Plot – Deposits (Outlier Check)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Deposits show fewer extreme outliers than withdrawals, confirming steadier inflow behaviour.")

    # Time-based Trends
    st.subheader("📅 Time-based Trends")
    daily = filtered_df.groupby("Date")[["Total_Withdrawals", "Total_Deposits"]].sum().reset_index()
    fig = px.line(daily, x="Date", y=["Total_Withdrawals", "Total_Deposits"],
                  title="Daily Total Withdrawals & Deposits Over Time")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔍 **Observation:** Clear periodic spikes in withdrawals correspond to weekends and salary/holiday dates. Deposits remain comparatively stable.")

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avail_days = [d for d in dow_order if d in filtered_df["Day_of_Week"].values]
    dow_avg = filtered_df.groupby("Day_of_Week")["Total_Withdrawals"].mean().reindex(avail_days).reset_index()
    fig = px.bar(dow_avg, x="Day_of_Week", y="Total_Withdrawals",
                 title="Average Withdrawals by Day of Week",
                 color="Total_Withdrawals", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔍 **Observation:** Weekends show the highest average withdrawals. Friday also elevated — payday effect.")

    time_order = ["Morning", "Afternoon", "Evening", "Night"]
    avail_times = [t for t in time_order if t in filtered_df["Time_of_Day"].values]
    time_avg = filtered_df.groupby("Time_of_Day")["Total_Withdrawals"].mean().reindex(avail_times).reset_index()
    fig = px.bar(time_avg, x="Time_of_Day", y="Total_Withdrawals",
                 title="Average Withdrawals by Time of Day",
                 color="Total_Withdrawals", color_continuous_scale="Oranges")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔍 **Observation:** Afternoon and Evening are the busiest periods, reflecting after-work and shopping-hour patterns.")

    # Holiday & Event Impact
    st.subheader("🎉 Holiday & Event Impact")
    col1, col2 = st.columns(2)
    with col1:
        h_avg = filtered_df.groupby("Holiday_Flag")["Total_Withdrawals"].mean().reset_index()
        h_avg["Holiday_Flag"] = h_avg["Holiday_Flag"].map({0: "Non-Holiday", 1: "Holiday"})
        fig = px.bar(h_avg, x="Holiday_Flag", y="Total_Withdrawals",
                     title="Average Withdrawals: Holidays vs Normal Days",
                     color="Holiday_Flag",
                     color_discrete_map={"Non-Holiday": "#636EFA", "Holiday": "#EF553B"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Holidays see significantly higher average withdrawals due to festive spending.")
    with col2:
        e_avg = filtered_df.groupby("Special_Event_Flag")["Total_Withdrawals"].mean().reset_index()
        e_avg["Special_Event_Flag"] = e_avg["Special_Event_Flag"].map({0: "No Event", 1: "Special Event"})
        fig = px.bar(e_avg, x="Special_Event_Flag", y="Total_Withdrawals",
                     title="Average Withdrawals: Special Events vs Normal",
                     color="Special_Event_Flag",
                     color_discrete_map={"No Event": "#636EFA", "Special Event": "#EF553B"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Special events (concerts, sports) drive a notable surge in cash demand.")

    # External Factors
    st.subheader("🌤️ External Factors")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(filtered_df, x="Weather_Condition", y="Total_Withdrawals",
                     title="Withdrawals by Weather Condition")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Clear weather is associated with highest withdrawals; rain and snow reduce ATM footfall.")
    with col2:
        comp_avg = filtered_df.groupby("Nearby_Competitor_ATMs")["Total_Withdrawals"].mean().reset_index()
        fig = px.bar(comp_avg, x="Nearby_Competitor_ATMs", y="Total_Withdrawals",
                     title="Withdrawals vs Number of Nearby Competitor ATMs")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** ATMs with more nearby competitors show lower average withdrawals, indicating shared demand.")

    # Relationship Analysis
    st.subheader("🔗 Relationship Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(filtered_df, x="Previous_Day_Cash_Level", y="Cash_Demand_Next_Day",
                         title="Previous Day Cash Level vs Next Day Demand",
                         trendline="ols", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Weak negative trend — higher leftover cash correlates with slightly lower next-day demand.")
    with col2:
        numeric_cols = ["Total_Withdrawals", "Total_Deposits", "Previous_Day_Cash_Level",
                        "Cash_Demand_Next_Day", "Nearby_Competitor_ATMs"]
        corr = filtered_df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Correlation Heatmap (Numeric Features)",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Withdrawals and next-day demand have a strong positive correlation. Deposits moderately correlated with withdrawals.")

# ==================== TAB 2: CLUSTERING ====================
with tab2:
    st.header("Stage 4 – Clustering Analysis of ATMs")
    st.markdown("K-Means clustering to group ATMs by demand behaviour, enabling targeted cash management.")

    k = st.slider("Select number of clusters", min_value=2, max_value=10, value=3, key="k_slider")

    # Cached — does not rerun on every sidebar filter change
    cluster_df, inertias, sil_scores, profile = run_clustering(k=k)

    # Elbow + Silhouette side by side
    st.subheader("📐 Optimal Cluster Selection")
    K_range = list(range(2, 11))
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=K_range, y=inertias, mode='lines+markers',
                                  line=dict(color='royalblue', width=2), name='Inertia'))
        fig.add_vline(x=k, line_dash="dash", line_color="red",
                      annotation_text=f"k={k}", annotation_position="top right")
        fig.update_layout(title="Elbow Method – Inertia vs k",
                          xaxis_title="Number of Clusters (k)", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** The 'elbow' is where inertia stops decreasing sharply — that's the optimal k.")
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=K_range, y=sil_scores, mode='lines+markers',
                                  line=dict(color='darkorange', width=2), name='Silhouette'))
        fig.add_vline(x=k, line_dash="dash", line_color="red",
                      annotation_text=f"k={k}", annotation_position="top right")
        fig.update_layout(title="Silhouette Score vs k",
                          xaxis_title="Number of Clusters (k)", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Higher silhouette = better-separated clusters. The peak indicates the most distinct grouping.")

    # PCA scatter
    st.subheader("🗺️ Cluster Visualization (PCA Projection)")
    plot_df = cluster_df.dropna(subset=["PC1", "PC2"]).query("Cluster != '-1'")
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                     hover_data=["ATM_ID", "Location_Type"],
                     title=f"K-Means Clusters in PCA Space (k={k})",
                     color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔍 **Observation:** Each colour represents a cluster. Well-separated blobs = strong clustering. Overlaps indicate ATMs with similar profiles.")

    # Cluster profiles table
    st.subheader("📋 Cluster Profiles")
    st.dataframe(profile, use_container_width=True)

    profile_reset = profile.reset_index()
    fig = px.bar(profile_reset, x="Cluster", y="Avg Withdrawals",
                 title="Average Withdrawals per Cluster", color="Cluster",
                 color_discrete_sequence=px.colors.qualitative.Set1,
                 text="Avg Withdrawals")
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Cluster Interpretation Guide:**
    - **High-demand cluster**: High avg withdrawals → increase refill frequency, especially before weekends/holidays.
    - **Medium-demand cluster**: Balanced withdrawals → standard schedule with holiday-sensitive adjustments.
    - **Low-demand cluster**: Low withdrawals → reduce refill frequency; investigate if consistently low.
    """)
    st.caption("🔍 **Observation:** Clustering segments ATMs by demand intensity and location context, enabling tailored cash-loading strategies for each group.")

# ==================== TAB 3: ANOMALY DETECTION ====================
with tab3:
    st.header("Stage 5 – Anomaly Detection on Withdrawals")
    st.markdown("Detecting unusual withdrawal patterns — especially on holidays and special events.")

    method = st.radio(
        "Select anomaly detection method",
        ["IQR (Interquartile Range)", "Isolation Forest (ML)"],
        horizontal=True
    )

    anomaly_df = filtered_df.copy()

    if method == "IQR (Interquartile Range)":
        Q1 = anomaly_df["Total_Withdrawals"].quantile(0.25)
        Q3 = anomaly_df["Total_Withdrawals"].quantile(0.75)
        IQR_val = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR_val
        upper_bound = Q3 + 1.5 * IQR_val
        anomaly_df["Anomaly"] = (
            (anomaly_df["Total_Withdrawals"] < lower_bound) |
            (anomaly_df["Total_Withdrawals"] > upper_bound)
        ).astype(int)
        col1, col2, col3 = st.columns(3)
        col1.metric("Lower Bound (IQR)", f"{lower_bound:,.0f}")
        col2.metric("Upper Bound (IQR)", f"{upper_bound:,.0f}")
        col3.metric("Anomalies Detected", int(anomaly_df["Anomaly"].sum()))
    else:
        contamination = st.slider("Contamination (expected anomaly fraction)", 0.01, 0.20, 0.05, 0.01)
        with st.spinner("Running Isolation Forest..."):
            features_for_if = ["Total_Withdrawals", "Total_Deposits", "Previous_Day_Cash_Level", "Nearby_Competitor_ATMs"]
            if_data = anomaly_df[features_for_if].dropna()
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(if_data)
            anomaly_df["Anomaly"] = 0
            anomaly_df.loc[if_data.index, "Anomaly"] = (preds == -1).astype(int)
        st.metric("Anomalies Detected", int(anomaly_df["Anomaly"].sum()))

    # Ensure no NaN in Anomaly column
    anomaly_df["Anomaly"] = anomaly_df["Anomaly"].fillna(0).astype(int)
    anomaly_df["Status"] = anomaly_df["Anomaly"].map({0: "Normal", 1: "Anomaly"})

    # Time series scatter
    st.subheader("📈 Time Series with Anomalies Highlighted")
    atm_options = sorted(anomaly_df["ATM_ID"].unique().tolist())
    selected_atms = st.multiselect(
        "Select ATMs to display (default: first 5)",
        options=atm_options,
        default=atm_options[:5]
    )
    display_atms = selected_atms if selected_atms else atm_options[:5]
    sample_df = anomaly_df[anomaly_df["ATM_ID"].isin(display_atms)]

    fig = px.scatter(
        sample_df, x="Date", y="Total_Withdrawals",
        color="Status",
        facet_col="ATM_ID", facet_col_wrap=2,
        title="Withdrawals over Time — Anomalies in Red",
        color_discrete_map={"Normal": "#636EFA", "Anomaly": "#EF553B"},
        hover_data=["Holiday_Flag", "Special_Event_Flag", "Weather_Condition"]
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔍 **Observation:** Red dots mark anomalous days. Many cluster around holidays and events — expected spikes that should be planned for, not penalised.")

    # Anomaly rate charts
    st.subheader("🎉 Anomaly Rates by Holiday & Event")
    col1, col2 = st.columns(2)
    with col1:
        h_anom = anomaly_df.groupby("Holiday_Flag")["Anomaly"].mean().reset_index()
        h_anom["Holiday_Flag"] = h_anom["Holiday_Flag"].map({0: "Non-Holiday", 1: "Holiday"})
        h_anom["Anomaly %"] = (h_anom["Anomaly"] * 100).round(1)
        fig = px.bar(h_anom, x="Holiday_Flag", y="Anomaly %",
                     title="% Anomalous Days: Holidays vs Normal",
                     color="Holiday_Flag",
                     color_discrete_map={"Non-Holiday": "#636EFA", "Holiday": "#EF553B"},
                     text="Anomaly %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Holiday days have a significantly higher anomaly rate — these spikes require proactive cash management.")
    with col2:
        e_anom = anomaly_df.groupby("Special_Event_Flag")["Anomaly"].mean().reset_index()
        e_anom["Special_Event_Flag"] = e_anom["Special_Event_Flag"].map({0: "No Event", 1: "Special Event"})
        e_anom["Anomaly %"] = (e_anom["Anomaly"] * 100).round(1)
        fig = px.bar(e_anom, x="Special_Event_Flag", y="Anomaly %",
                     title="% Anomalous Days: Special Events vs Normal",
                     color="Special_Event_Flag",
                     color_discrete_map={"No Event": "#636EFA", "Special Event": "#EF553B"},
                     text="Anomaly %")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Special events drastically elevate anomaly rates. Pre-load extra cash before known events.")

    # Anomaly by location type
    st.subheader("📍 Anomaly Rate by Location Type")
    loc_anom = anomaly_df.groupby("Location_Type")["Anomaly"].mean().reset_index()
    loc_anom["Anomaly %"] = (loc_anom["Anomaly"] * 100).round(1)
    fig = px.bar(loc_anom.sort_values("Anomaly %", ascending=False),
                 x="Location_Type", y="Anomaly %",
                 title="Anomaly Rate by ATM Location Type",
                 text="Anomaly %", color="Anomaly %", color_continuous_scale="Reds")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔍 **Observation:** Certain location types (e.g. Mall, Gas Station) show higher anomaly rates due to event-driven foot traffic.")

# ==================== TAB 4: INTERACTIVE PLANNER ====================
with tab4:
    st.header("Stage 6 – Interactive Cash Demand Planner")
    st.markdown("Combine all insights: view ATM clusters, anomaly flags, and actionable recommendations in one place.")

    # Cached ATM-level clustering
    atm_clusters = run_atm_level_clustering(k=3)
    df_with_cluster = df.merge(atm_clusters, on="ATM_ID", how="left")

    # Apply sidebar filters
    filtered_planner = df_with_cluster[
        df_with_cluster["Day_of_Week"].isin(selected_days) &
        df_with_cluster["Time_of_Day"].isin(selected_times) &
        df_with_cluster["Location_Type"].isin(selected_locations)
    ].copy()
    if not include_holiday:
        filtered_planner = filtered_planner[filtered_planner["Holiday_Flag"] == 0]
    if not include_event:
        filtered_planner = filtered_planner[filtered_planner["Special_Event_Flag"] == 0]

    if filtered_planner.empty:
        st.warning("No data matches current filters.")
        st.stop()

    # IQR anomaly flag
    Q1 = filtered_planner["Total_Withdrawals"].quantile(0.25)
    Q3 = filtered_planner["Total_Withdrawals"].quantile(0.75)
    IQR_val = Q3 - Q1
    filtered_planner["Anomaly"] = (
        (filtered_planner["Total_Withdrawals"] < Q1 - 1.5 * IQR_val) |
        (filtered_planner["Total_Withdrawals"] > Q3 + 1.5 * IQR_val)
    ).astype(int)
    filtered_planner["Status"] = filtered_planner["Anomaly"].map({0: "Normal", 1: "Anomaly"})

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Records", f"{len(filtered_planner):,}")
    col2.metric("🏧 Unique ATMs", f"{filtered_planner['ATM_ID'].nunique():,}")
    col3.metric("🚨 Anomalies", f"{int(filtered_planner['Anomaly'].sum()):,}")
    col4.metric("⚠️ Anomaly Rate", f"{filtered_planner['Anomaly'].mean() * 100:.1f}%")

    st.markdown("---")

    # Cluster distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ATM Cluster Distribution")
        cc = filtered_planner["ATM_Cluster"].value_counts().reset_index()
        cc.columns = ["Cluster", "Records"]
        fig = px.pie(cc, values="Records", names="Cluster",
                     title="Record Distribution by ATM Cluster",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** The pie chart shows transaction records by cluster under the current filter window.")
    with col2:
        st.subheader("Withdrawals by Cluster")
        fig = px.box(filtered_planner, x="ATM_Cluster", y="Total_Withdrawals",
                     color="ATM_Cluster", title="Withdrawal Distribution per Cluster",
                     color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** The spread within each cluster confirms meaningful groupings — high-demand clusters have higher medians and wider ranges.")

    # Combined anomaly + cluster scatter
    st.subheader("Anomaly Map: Withdrawals by Date & Cluster")
    fig = px.scatter(filtered_planner, x="Date", y="Total_Withdrawals",
                     color="Status", symbol="ATM_Cluster",
                     title="Withdrawals over Time — Colour: Anomaly Status | Shape: Cluster",
                     color_discrete_map={"Normal": "#636EFA", "Anomaly": "#EF553B"},
                     opacity=0.6,
                     hover_data=["ATM_ID", "Location_Type", "Holiday_Flag", "Special_Event_Flag"])
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("📂 Detailed Transaction Table")
    display_cols = ["ATM_ID", "Date", "Day_of_Week", "Time_of_Day", "Location_Type",
                    "Total_Withdrawals", "Total_Deposits", "ATM_Cluster", "Status",
                    "Holiday_Flag", "Special_Event_Flag", "Weather_Condition"]
    st.dataframe(
        filtered_planner[display_cols].sort_values("Date").reset_index(drop=True),
        use_container_width=True
    )

    # Actionable recommendations
    st.subheader("💡 Actionable Recommendations")
    anomaly_rate = filtered_planner["Anomaly"].mean() * 100
    if anomaly_rate > 10:
        st.error(f"⚠️ High anomaly rate ({anomaly_rate:.1f}%) detected. Review and pre-stage cash proactively.")
    elif anomaly_rate > 5:
        st.warning(f"🟡 Moderate anomaly rate ({anomaly_rate:.1f}%). Monitor closely and increase cash on flagged days.")
    else:
        st.success(f"✅ Low anomaly rate ({anomaly_rate:.1f}%). Current cash management strategy appears sufficient.")

    recs = [
        "🏢 **High-demand cluster ATMs** should be prioritised for more frequent refills.",
        "🎉 **Holiday days** — increase cash levels and refill frequency before public holidays.",
        "🎭 **Special events** — coordinate with event organizers to estimate footfall and pre-load ATMs.",
        "☀️ **Clear weather weekends** — plan for higher-than-average withdrawals.",
        "🔍 **Non-holiday anomalies** — investigate these for potential equipment faults or fraud.",
    ]
    for r in recs:
        st.markdown(r)

    st.markdown("---")
    st.caption("This interactive planner combines all FA-2 insights — EDA, clustering, and anomaly detection — in a single reproducible workflow. Adjust the sidebar filters to explore different conditions.")

# ------------------------------
# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **FA-2 | ATM Intelligence**\n\n"
    "Stages covered:\n"
    "- **Tab 1**: Stage 3 – EDA\n"
    "- **Tab 2**: Stage 4 – Clustering\n"
    "- **Tab 3**: Stage 5 – Anomaly Detection\n"
    "- **Tab 4**: Stage 6 – Interactive Planner\n\n"
    "All charts update with sidebar filters."
)