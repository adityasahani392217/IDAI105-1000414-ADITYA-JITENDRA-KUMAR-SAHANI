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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ATM Demand Intelligence",
    layout="wide",
    page_icon="🏧"
)
st.title("🏧 ATM Demand Forecasting & Insights")
st.markdown("Interactive dashboard for exploratory analysis, clustering, and anomaly detection.")

# ─────────────────────────────────────────────
# Location mapping: original → Urban / Semi-Urban / Rural
# ─────────────────────────────────────────────
LOCATION_MAP = {
    "Bank Branch":  "Urban",
    "Mall":         "Urban",
    "Supermarket":  "Semi-Urban",
    "Standalone":   "Semi-Urban",
    "Gas Station":  "Rural",
}

# ─────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("atm_cash_management_dataset.csv")
    df["Date"]       = pd.to_datetime(df["Date"])
    df["Month"]      = df["Date"].dt.month
    df["Year"]       = df["Date"].dt.year
    df["Is_Weekend"] = df["Day_of_Week"].isin(["Saturday", "Sunday"]).astype(int)
    # Map location types → Urban / Semi-Urban / Rural
    df["Location_Type"] = df["Location_Type"].map(LOCATION_MAP).fillna("Semi-Urban")
    return df

try:
    df = load_data()
    st.success("✅ Loaded dataset: atm_cash_management_dataset.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found. Place 'atm_cash_management_dataset.csv' in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Pre-computed heavy operations (cached)
# ─────────────────────────────────────────────
@st.cache_data
def run_clustering(k: int = 3):
    """
    Transaction-level K-Means clustering.
    Features: Total_Withdrawals, Total_Deposits, Nearby_Competitor_ATMs, Location_Encoded
    Also computes Elbow + Silhouette curves and PCA projection.
    """
    cluster_df = df.copy()
    le = LabelEncoder()
    cluster_df["Location_Encoded"] = le.fit_transform(cluster_df["Location_Type"])

    feature_cols = [
        "Total_Withdrawals",
        "Total_Deposits",
        "Nearby_Competitor_ATMs",
        "Location_Encoded",
        "Is_Weekend",
        "Holiday_Flag",
        "Special_Event_Flag",
    ]
    X = cluster_df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow + silhouette for k = 2 … 10
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

    # PCA 2-D projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_var = pca.explained_variance_ratio_
    cluster_df["PC1"] = np.nan
    cluster_df["PC2"] = np.nan
    cluster_df.loc[X.index, "PC1"] = X_pca[:, 0]
    cluster_df.loc[X.index, "PC2"] = X_pca[:, 1]
    cluster_df["Cluster"] = cluster_df["Cluster"].fillna(-1).astype(int).astype(str)

    # Cluster profile
    valid = cluster_df[cluster_df["Cluster"] != "-1"]
    profile_num = valid.groupby("Cluster")[
        ["Total_Withdrawals", "Total_Deposits", "Nearby_Competitor_ATMs"]
    ].mean().round(1)
    profile_loc = valid.groupby("Cluster")["Location_Type"].agg(
        lambda x: x.mode()[0] if len(x) > 0 else "Unknown"
    )
    profile = profile_num.copy()
    profile["Top Location"] = profile_loc
    profile = profile.rename(columns={
        "Total_Withdrawals":      "Avg Withdrawals",
        "Total_Deposits":         "Avg Deposits",
        "Nearby_Competitor_ATMs": "Avg Competitors",
    })

    return cluster_df, inertias, sil_scores, profile, pca_var


@st.cache_data
def run_atm_level_clustering(k: int = 3):
    """
    ATM-level K-Means: cluster each ATM by its mean behaviour.
    Used in the Interactive Planner tab.
    """
    atm_agg = df.groupby("ATM_ID").agg(
        Total_Withdrawals     = ("Total_Withdrawals",      "mean"),
        Total_Deposits        = ("Total_Deposits",         "mean"),
        Nearby_Competitor_ATMs= ("Nearby_Competitor_ATMs", "first"),
        Holiday_Flag          = ("Holiday_Flag",           "mean"),
        Special_Event_Flag    = ("Special_Event_Flag",     "mean"),
        Is_Weekend            = ("Is_Weekend",             "mean"),
        Location_Type         = ("Location_Type",          "first"),
    ).reset_index()

    le = LabelEncoder()
    atm_agg["Location_Encoded"] = le.fit_transform(atm_agg["Location_Type"])

    X = atm_agg[[
        "Total_Withdrawals", "Total_Deposits",
        "Nearby_Competitor_ATMs", "Location_Encoded",
        "Holiday_Flag", "Special_Event_Flag", "Is_Weekend",
    ]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    atm_agg["ATM_Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    return atm_agg[["ATM_ID", "ATM_Cluster"]]


# ─────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────
st.sidebar.header("🔍 Interactive Filters")

selected_days = st.sidebar.multiselect(
    "Day of Week",
    options=df["Day_of_Week"].unique().tolist(),
    default=df["Day_of_Week"].unique().tolist(),
)
selected_times = st.sidebar.multiselect(
    "Time of Day",
    options=df["Time_of_Day"].unique().tolist(),
    default=df["Time_of_Day"].unique().tolist(),
)
# Now only Urban / Semi-Urban / Rural appear here
location_types = sorted(df["Location_Type"].unique().tolist())   # Urban, Rural, Semi-Urban
selected_locations = st.sidebar.multiselect(
    "Location Type",
    options=location_types,
    default=location_types,
)
include_holiday = st.sidebar.checkbox("Include Holidays",       value=True)
include_event   = st.sidebar.checkbox("Include Special Events", value=True)

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

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Data Analysis",
    "📈 Clustering ATMs",
    "🚨 Anomaly Detection",
    "⚙️ Interactive Planner",
])

# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    st.header("Stage 3 – Exploratory Data Analysis")
    st.markdown("Visual exploration to uncover trends, patterns, and relationships in ATM cash demand data.")

    # ── Distribution Analysis ──────────────────
    st.subheader("📦 Distribution Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            filtered_df, x="Total_Withdrawals", nbins=50, marginal="box",
            title="Histogram of Total Withdrawals",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "🔍 **Observation:** Withdrawals are right-skewed; most days see moderate demand "
            "with a long tail of high-demand events (paydays, holidays)."
        )
    with col2:
        fig = px.histogram(
            filtered_df, x="Total_Deposits", nbins=50, marginal="box",
            title="Histogram of Total Deposits",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "🔍 **Observation:** Deposits are also skewed and typically lower than withdrawals, "
            "indicating net cash outflow at ATMs."
        )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(filtered_df, y="Total_Withdrawals", title="Box Plot – Withdrawals (Outlier Check)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Several high-value outliers present; likely correspond to holiday or event days.")
    with col2:
        fig = px.box(filtered_df, y="Total_Deposits", title="Box Plot – Deposits (Outlier Check)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Deposits show fewer extreme outliers than withdrawals, confirming steadier inflow behaviour.")

    # ── Time-based Trends ─────────────────────
    st.subheader("📅 Time-based Trends")
    daily = filtered_df.groupby("Date")[["Total_Withdrawals", "Total_Deposits"]].sum().reset_index()
    fig = px.line(daily, x="Date", y=["Total_Withdrawals", "Total_Deposits"],
                  title="Daily Total Withdrawals & Deposits Over Time")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔍 **Observation:** Clear periodic spikes in withdrawals correspond to weekends and "
        "salary/holiday dates. Deposits remain comparatively stable."
    )

    dow_order   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avail_days  = [d for d in dow_order if d in filtered_df["Day_of_Week"].values]
    dow_avg = (
        filtered_df.groupby("Day_of_Week")["Total_Withdrawals"]
        .mean()
        .reindex(avail_days)
        .reset_index()
    )
    fig = px.bar(
        dow_avg, x="Day_of_Week", y="Total_Withdrawals",
        title="Average Withdrawals by Day of Week",
        color="Total_Withdrawals", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔍 **Observation:** Weekends show the highest average withdrawals. Friday also elevated — payday effect.")

    time_order  = ["Morning", "Afternoon", "Evening", "Night"]
    avail_times = [t for t in time_order if t in filtered_df["Time_of_Day"].values]
    time_avg = (
        filtered_df.groupby("Time_of_Day")["Total_Withdrawals"]
        .mean()
        .reindex(avail_times)
        .reset_index()
    )
    fig = px.bar(
        time_avg, x="Time_of_Day", y="Total_Withdrawals",
        title="Average Withdrawals by Time of Day",
        color="Total_Withdrawals", color_continuous_scale="Oranges",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔍 **Observation:** Afternoon and Evening are the busiest periods, "
        "reflecting after-work and shopping-hour patterns."
    )

    # ── Location Type Breakdown ───────────────
    st.subheader("📍 Withdrawals by Location Type (Urban / Semi-Urban / Rural)")
    loc_avg = (
        filtered_df.groupby("Location_Type")["Total_Withdrawals"]
        .mean()
        .reset_index()
        .sort_values("Total_Withdrawals", ascending=False)
    )
    fig = px.bar(
        loc_avg, x="Location_Type", y="Total_Withdrawals",
        title="Average Withdrawals by Location Type",
        color="Location_Type",
        color_discrete_map={
            "Urban":      "#EF553B",
            "Semi-Urban": "#636EFA",
            "Rural":      "#00CC96",
        },
        text="Total_Withdrawals",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔍 **Observation:** Urban ATMs (Bank Branches & Malls) record the highest average "
        "withdrawals; Rural ATMs (Gas Stations) see the lowest but most variable demand."
    )

    # ── Holiday & Event Impact ────────────────
    st.subheader("🎉 Holiday & Event Impact")
    col1, col2 = st.columns(2)
    with col1:
        h_avg = filtered_df.groupby("Holiday_Flag")["Total_Withdrawals"].mean().reset_index()
        h_avg["Holiday_Flag"] = h_avg["Holiday_Flag"].map({0: "Non-Holiday", 1: "Holiday"})
        fig = px.bar(
            h_avg, x="Holiday_Flag", y="Total_Withdrawals",
            title="Average Withdrawals: Holidays vs Normal Days",
            color="Holiday_Flag",
            color_discrete_map={"Non-Holiday": "#636EFA", "Holiday": "#EF553B"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Holidays see significantly higher average withdrawals due to festive spending.")
    with col2:
        e_avg = filtered_df.groupby("Special_Event_Flag")["Total_Withdrawals"].mean().reset_index()
        e_avg["Special_Event_Flag"] = e_avg["Special_Event_Flag"].map({0: "No Event", 1: "Special Event"})
        fig = px.bar(
            e_avg, x="Special_Event_Flag", y="Total_Withdrawals",
            title="Average Withdrawals: Special Events vs Normal",
            color="Special_Event_Flag",
            color_discrete_map={"No Event": "#636EFA", "Special Event": "#EF553B"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Special events (concerts, sports) drive a notable surge in cash demand.")

    # ── External Factors ──────────────────────
    st.subheader("🌤️ External Factors")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            filtered_df, x="Weather_Condition", y="Total_Withdrawals",
            title="Withdrawals by Weather Condition",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Clear weather is associated with highest withdrawals; rain and snow reduce ATM footfall.")
    with col2:
        comp_avg = (
            filtered_df.groupby("Nearby_Competitor_ATMs")["Total_Withdrawals"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            comp_avg, x="Nearby_Competitor_ATMs", y="Total_Withdrawals",
            title="Withdrawals vs Number of Nearby Competitor ATMs",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** ATMs with more nearby competitors show lower average withdrawals, indicating shared demand.")

    # ── Relationship Analysis ─────────────────
    st.subheader("🔗 Relationship Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            filtered_df, x="Previous_Day_Cash_Level", y="Cash_Demand_Next_Day",
            title="Previous Day Cash Level vs Next Day Demand",
            trendline="ols", opacity=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Weak negative trend — higher leftover cash correlates with slightly lower next-day demand.")
    with col2:
        numeric_cols = [
            "Total_Withdrawals", "Total_Deposits",
            "Previous_Day_Cash_Level", "Cash_Demand_Next_Day",
            "Nearby_Competitor_ATMs",
        ]
        corr = filtered_df[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            title="Correlation Heatmap (Numeric Features)",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "🔍 **Observation:** Withdrawals and next-day demand have a strong positive correlation. "
            "Deposits moderately correlated with withdrawals."
        )

# ══════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ══════════════════════════════════════════════
with tab2:
    st.header("Stage 4 – Clustering Analysis of ATMs")
    st.markdown(
        "K-Means clustering groups ATMs by demand behaviour (withdrawals, deposits, "
        "competitor count, location, holiday & weekend patterns), enabling targeted cash management."
    )

    k = st.slider("Select number of clusters", min_value=2, max_value=10, value=3, key="k_slider")

    cluster_df, inertias, sil_scores, profile, pca_var = run_clustering(k=k)

    # ── Optimal cluster selection ─────────────
    st.subheader("📐 Optimal Cluster Selection")
    K_range = list(range(2, 11))
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=K_range, y=inertias, mode="lines+markers",
            line=dict(color="royalblue", width=2), name="Inertia",
        ))
        fig.add_vline(x=k, line_dash="dash", line_color="red",
                      annotation_text=f"k={k}", annotation_position="top right")
        fig.update_layout(
            title="Elbow Method – Inertia vs k",
            xaxis_title="Number of Clusters (k)", yaxis_title="Inertia",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** The 'elbow' is where inertia stops decreasing sharply — that's the optimal k.")
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=K_range, y=sil_scores, mode="lines+markers",
            line=dict(color="darkorange", width=2), name="Silhouette",
        ))
        fig.add_vline(x=k, line_dash="dash", line_color="red",
                      annotation_text=f"k={k}", annotation_position="top right")
        fig.update_layout(
            title="Silhouette Score vs k",
            xaxis_title="Number of Clusters (k)", yaxis_title="Score",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Higher silhouette = better-separated clusters. The peak indicates the most distinct grouping.")

    # ── PCA scatter ───────────────────────────
    st.subheader("🗺️ Cluster Visualization (PCA Projection)")
    st.markdown(
        f"PCA variance explained — PC1: **{pca_var[0]*100:.1f}%** | PC2: **{pca_var[1]*100:.1f}%** "
        f"| Total: **{sum(pca_var)*100:.1f}%**"
    )
    plot_df = cluster_df.dropna(subset=["PC1", "PC2"]).query("Cluster != '-1'")
    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Cluster",
        hover_data=["ATM_ID", "Location_Type"],
        title=f"K-Means Clusters in PCA Space (k={k})",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔍 **Observation:** Each colour represents a cluster. Well-separated blobs = strong clustering. "
        "Overlaps indicate ATMs with similar demand profiles across different location zones."
    )

    # ── Cluster profiles ──────────────────────
    st.subheader("📋 Cluster Profiles")
    st.dataframe(profile, use_container_width=True)

    profile_reset = profile.reset_index()
    fig = px.bar(
        profile_reset, x="Cluster", y="Avg Withdrawals",
        title="Average Withdrawals per Cluster",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set1,
        text="Avg Withdrawals",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Cluster Interpretation Guide:**
- 🏙️ **Urban (High-demand) cluster** — Bank Branch & Mall ATMs; highest avg withdrawals → increase refill frequency before weekends/holidays.
- 🏘️ **Semi-Urban (Steady-demand) cluster** — Supermarket & Standalone ATMs; balanced withdrawals → standard schedule with holiday-sensitive adjustments.
- 🌾 **Rural (Low-demand) cluster** — Gas Station ATMs; lowest withdrawals → reduce refill frequency; investigate if consistently below threshold.
    """)
    st.caption(
        "🔍 **Observation:** Clustering by location zone (Urban/Semi-Urban/Rural) alongside demand "
        "features segments ATMs into operationally meaningful groups for tailored cash-loading strategies."
    )

# ══════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab3:
    st.header("Stage 5 – Anomaly Detection on Withdrawals")
    st.markdown("Detecting unusual withdrawal patterns — especially on holidays and special events.")

    method = st.radio(
        "Select anomaly detection method",
        [
            "IQR (Interquartile Range)",
            "Z-Score (Statistical)",
            "Isolation Forest (ML)",
        ],
        horizontal=True,
    )

    anomaly_df = filtered_df.copy()

    # ── IQR ──────────────────────────────────
    if method == "IQR (Interquartile Range)":
        Q1  = anomaly_df["Total_Withdrawals"].quantile(0.25)
        Q3  = anomaly_df["Total_Withdrawals"].quantile(0.75)
        IQR_val     = Q3 - Q1
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

    # ── Z-Score ───────────────────────────────
    elif method == "Z-Score (Statistical)":
        z_thresh = st.slider("Z-Score threshold", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
        z_scores = np.abs(stats.zscore(anomaly_df["Total_Withdrawals"].dropna()))
        anomaly_df["Z_Score"] = np.nan
        anomaly_df.loc[anomaly_df["Total_Withdrawals"].notna(), "Z_Score"] = z_scores
        anomaly_df["Anomaly"] = (anomaly_df["Z_Score"] > z_thresh).astype(int)
        col1, col2 = st.columns(2)
        col1.metric("Z-Score Threshold", f"± {z_thresh:.1f}")
        col2.metric("Anomalies Detected", int(anomaly_df["Anomaly"].sum()))
        # Z-score distribution chart
        fig = px.histogram(
            anomaly_df.dropna(subset=["Z_Score"]),
            x="Z_Score", nbins=60,
            title="Z-Score Distribution of Withdrawals",
            color_discrete_sequence=["#636EFA"],
        )
        fig.add_vline(x=z_thresh,  line_dash="dash", line_color="red",  annotation_text=f"+{z_thresh}")
        fig.add_vline(x=-z_thresh, line_dash="dash", line_color="red",  annotation_text=f"-{z_thresh}")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"🔍 **Observation:** Points beyond ±{z_thresh} standard deviations are flagged as anomalies.")

    # ── Isolation Forest ─────────────────────
    else:
        contamination = st.slider(
            "Contamination (expected anomaly fraction)",
            0.01, 0.20, 0.05, 0.01,
        )
        with st.spinner("Running Isolation Forest…"):
            features_for_if = [
                "Total_Withdrawals", "Total_Deposits",
                "Previous_Day_Cash_Level", "Nearby_Competitor_ATMs",
                "Holiday_Flag", "Special_Event_Flag", "Is_Weekend",
            ]
            if_data = anomaly_df[features_for_if].dropna()
            iso   = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
            preds = iso.fit_predict(if_data)
            anomaly_df["Anomaly"] = 0
            anomaly_df.loc[if_data.index, "Anomaly"] = (preds == -1).astype(int)
        st.metric("Anomalies Detected (Isolation Forest)", int(anomaly_df["Anomaly"].sum()))

    # Ensure clean Anomaly column
    anomaly_df["Anomaly"] = anomaly_df["Anomaly"].fillna(0).astype(int)
    anomaly_df["Status"]  = anomaly_df["Anomaly"].map({0: "Normal", 1: "Anomaly"})

    # ── Time series scatter ───────────────────
    st.subheader("📈 Time Series with Anomalies Highlighted")
    atm_options   = sorted(anomaly_df["ATM_ID"].unique().tolist())
    selected_atms = st.multiselect(
        "Select ATMs to display (default: first 5)",
        options=atm_options,
        default=atm_options[:5],
    )
    display_atms = selected_atms if selected_atms else atm_options[:5]
    sample_df    = anomaly_df[anomaly_df["ATM_ID"].isin(display_atms)]

    fig = px.scatter(
        sample_df, x="Date", y="Total_Withdrawals",
        color="Status",
        facet_col="ATM_ID", facet_col_wrap=2,
        title="Withdrawals over Time — Anomalies in Red",
        color_discrete_map={"Normal": "#636EFA", "Anomaly": "#EF553B"},
        hover_data=["Holiday_Flag", "Special_Event_Flag", "Weather_Condition", "Location_Type"],
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔍 **Observation:** Red dots mark anomalous days. Many cluster around holidays and events — "
        "expected spikes that should be planned for, not penalised."
    )

    # ── Anomaly rates by holiday & event ─────
    st.subheader("🎉 Anomaly Rates by Holiday & Event")
    col1, col2 = st.columns(2)
    with col1:
        h_anom = anomaly_df.groupby("Holiday_Flag")["Anomaly"].mean().reset_index()
        h_anom["Holiday_Flag"] = h_anom["Holiday_Flag"].map({0: "Non-Holiday", 1: "Holiday"})
        h_anom["Anomaly %"]    = (h_anom["Anomaly"] * 100).round(1)
        fig = px.bar(
            h_anom, x="Holiday_Flag", y="Anomaly %",
            title="% Anomalous Days: Holidays vs Normal",
            color="Holiday_Flag",
            color_discrete_map={"Non-Holiday": "#636EFA", "Holiday": "#EF553B"},
            text="Anomaly %",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Holiday days have a significantly higher anomaly rate — these spikes require proactive cash management.")
    with col2:
        e_anom = anomaly_df.groupby("Special_Event_Flag")["Anomaly"].mean().reset_index()
        e_anom["Special_Event_Flag"] = e_anom["Special_Event_Flag"].map({0: "No Event", 1: "Special Event"})
        e_anom["Anomaly %"]          = (e_anom["Anomaly"] * 100).round(1)
        fig = px.bar(
            e_anom, x="Special_Event_Flag", y="Anomaly %",
            title="% Anomalous Days: Special Events vs Normal",
            color="Special_Event_Flag",
            color_discrete_map={"No Event": "#636EFA", "Special Event": "#EF553B"},
            text="Anomaly %",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** Special events drastically elevate anomaly rates. Pre-load extra cash before known events.")

    # ── Anomaly by location type ──────────────
    st.subheader("📍 Anomaly Rate by Location Type (Urban / Semi-Urban / Rural)")
    loc_anom = anomaly_df.groupby("Location_Type")["Anomaly"].mean().reset_index()
    loc_anom["Anomaly %"] = (loc_anom["Anomaly"] * 100).round(1)
    fig = px.bar(
        loc_anom.sort_values("Anomaly %", ascending=False),
        x="Location_Type", y="Anomaly %",
        title="Anomaly Rate by ATM Location Zone",
        text="Anomaly %",
        color="Location_Type",
        color_discrete_map={
            "Urban":      "#EF553B",
            "Semi-Urban": "#FFA15A",
            "Rural":      "#FECB52",
        },
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔍 **Observation:** Urban ATMs show the highest anomaly rates due to event-driven "
        "and holiday foot traffic; Rural ATMs are more predictable."
    )

# ══════════════════════════════════════════════
# TAB 4 — INTERACTIVE PLANNER
# ══════════════════════════════════════════════
with tab4:
    st.header("Stage 6 – Interactive Cash Demand Planner")
    st.markdown("Combine all insights: view ATM clusters, anomaly flags, and actionable recommendations in one place.")

    # Cached ATM-level clustering
    atm_clusters    = run_atm_level_clustering(k=3)
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
    Q1  = filtered_planner["Total_Withdrawals"].quantile(0.25)
    Q3  = filtered_planner["Total_Withdrawals"].quantile(0.75)
    IQR_val = Q3 - Q1
    filtered_planner["Anomaly"] = (
        (filtered_planner["Total_Withdrawals"] < Q1 - 1.5 * IQR_val) |
        (filtered_planner["Total_Withdrawals"] > Q3 + 1.5 * IQR_val)
    ).astype(int)
    filtered_planner["Status"] = filtered_planner["Anomaly"].map({0: "Normal", 1: "Anomaly"})

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Records",      f"{len(filtered_planner):,}")
    col2.metric("🏧 Unique ATMs",  f"{filtered_planner['ATM_ID'].nunique():,}")
    col3.metric("🚨 Anomalies",    f"{int(filtered_planner['Anomaly'].sum()):,}")
    col4.metric("⚠️ Anomaly Rate", f"{filtered_planner['Anomaly'].mean() * 100:.1f}%")
    st.markdown("---")

    # Cluster distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ATM Cluster Distribution")
        cc = filtered_planner["ATM_Cluster"].value_counts().reset_index()
        cc.columns = ["Cluster", "Records"]
        fig = px.pie(
            cc, values="Records", names="Cluster",
            title="Record Distribution by ATM Cluster",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("🔍 **Observation:** The pie chart shows transaction records by cluster under the current filter window.")
    with col2:
        st.subheader("Withdrawals by Cluster")
        fig = px.box(
            filtered_planner, x="ATM_Cluster", y="Total_Withdrawals",
            color="ATM_Cluster",
            title="Withdrawal Distribution per Cluster",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "🔍 **Observation:** The spread within each cluster confirms meaningful groupings — "
            "high-demand clusters have higher medians and wider ranges."
        )

    # Withdrawals by Location Zone in planner
    st.subheader("📍 Withdrawal Distribution by Location Zone")
    fig = px.box(
        filtered_planner, x="Location_Type", y="Total_Withdrawals",
        color="Location_Type",
        title="Withdrawal Range: Urban vs Semi-Urban vs Rural",
        color_discrete_map={
            "Urban":      "#EF553B",
            "Semi-Urban": "#636EFA",
            "Rural":      "#00CC96",
        },
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔍 **Observation:** Urban ATMs have the highest median and widest withdrawal range; "
        "Rural ATMs are lower and more consistent."
    )

    # Combined anomaly + cluster scatter
    st.subheader("Anomaly Map: Withdrawals by Date & Cluster")
    fig = px.scatter(
        filtered_planner, x="Date", y="Total_Withdrawals",
        color="Status", symbol="ATM_Cluster",
        title="Withdrawals over Time — Colour: Anomaly Status | Shape: Cluster",
        color_discrete_map={"Normal": "#636EFA", "Anomaly": "#EF553B"},
        opacity=0.6,
        hover_data=["ATM_ID", "Location_Type", "Holiday_Flag", "Special_Event_Flag"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed transaction table
    st.subheader("📂 Detailed Transaction Table")
    display_cols = [
        "ATM_ID", "Date", "Day_of_Week", "Time_of_Day", "Location_Type",
        "Total_Withdrawals", "Total_Deposits", "ATM_Cluster", "Status",
        "Holiday_Flag", "Special_Event_Flag", "Weather_Condition",
    ]
    st.dataframe(
        filtered_planner[display_cols].sort_values("Date").reset_index(drop=True),
        use_container_width=True,
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
        "🏙️ **Urban ATMs** (Bank Branch & Mall zones) — highest demand; prioritise for frequent refills and pre-holiday loading.",
        "🏘️ **Semi-Urban ATMs** (Supermarket & Standalone zones) — standard refill schedule with adjustments around events.",
        "🌾 **Rural ATMs** (Gas Station zones) — lower demand but more volatile; consider demand-triggered refills.",
        "🎉 **Holiday days** — increase cash levels and refill frequency at least one day in advance.",
        "🎭 **Special events** — coordinate with event organisers to estimate footfall and pre-load nearby ATMs.",
        "☀️ **Clear weather weekends** — plan for higher-than-average withdrawals.",
        "🔍 **Non-holiday anomalies** — investigate these for potential equipment faults or fraud.",
    ]
    for r in recs:
        st.markdown(r)

    st.markdown("---")
    st.caption(
        "This interactive planner combines all FA-2 insights — EDA, clustering, and anomaly detection — "
        "in a single reproducible workflow. Adjust the sidebar filters to explore different conditions."
    )

# ─────────────────────────────────────────────
# Sidebar footer
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **FA-2 | ATM Intelligence**\n\n"
    "Stages covered:\n"
    "- **Tab 1**: Stage 3 – EDA\n"
    "- **Tab 2**: Stage 4 – Clustering\n"
    "- **Tab 3**: Stage 5 – Anomaly Detection\n"
    "- **Tab 4**: Stage 6 – Interactive Planner\n\n"
    "Location zones: Urban · Semi-Urban · Rural\n"
    "All charts update with sidebar filters."
)
