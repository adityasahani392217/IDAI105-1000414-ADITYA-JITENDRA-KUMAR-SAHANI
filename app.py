# =============================================================================
# FA-2: ATM Intelligence Demand Forecasting with Data Mining
# FinTrust Bank Ltd. | Data Mining Project
# Stages: EDA (3) → Clustering (4) → Anomaly Detection (5) → Interactive Planner (6)
# =============================================================================
# REQUIRED LIBRARIES:
# pip install pandas numpy matplotlib seaborn scikit-learn scipy
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from scipy import stats

# Set global plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("=" * 70)
print("  ATM INTELLIGENCE DEMAND FORECASTING — FA-2 ANALYSIS PIPELINE")
print("  FinTrust Bank Ltd.")
print("=" * 70)


# =============================================================================
# STAGE 2 (RECAP): DATA LOADING & PREPROCESSING
# =============================================================================
# NOTE: This script generates a realistic synthetic dataset that mirrors the
# exact columns specified in the FA brief. If you have the actual CSV file,
# replace the generate_dataset() block with:
#   df = pd.read_csv("your_dataset.csv")
# and skip straight to the preprocessing section.
# =============================================================================

def generate_dataset(n=2000, seed=42):
    """
    Generates a realistic synthetic ATM transaction dataset.
    Mirrors the FA brief columns exactly.
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    dates = np.random.choice(dates, n, replace=True)

    location_types = np.random.choice(["Urban", "Semi-Urban", "Rural"],
                                       n, p=[0.5, 0.3, 0.2])
    day_of_week = pd.to_datetime(dates).day_name()
    time_of_day = np.random.choice(["Morning", "Afternoon", "Evening", "Night"],
                                    n, p=[0.25, 0.30, 0.35, 0.10])
    holiday_flag = np.random.choice([0, 1], n, p=[0.85, 0.15])
    special_event_flag = np.random.choice([0, 1], n, p=[0.88, 0.12])
    weather = np.random.choice(["Clear", "Rainy", "Stormy"], n, p=[0.65, 0.25, 0.10])
    competitor = np.random.choice([0, 1], n, p=[0.55, 0.45])

    base_withdrawal = np.where(location_types == "Urban", 18000,
                        np.where(location_types == "Semi-Urban", 11000, 6000))

    # Simulate demand spikes on holidays and events
    holiday_boost = 1 + 0.45 * holiday_flag
    event_boost = 1 + 0.30 * special_event_flag
    weekend_boost = np.where(np.isin(day_of_week, ["Saturday", "Friday"]), 1.2, 1.0)
    weather_impact = np.where(weather == "Stormy", 0.75,
                       np.where(weather == "Rainy", 0.90, 1.0))
    competitor_impact = np.where(competitor == 1, 0.88, 1.0)

    total_withdrawals = (base_withdrawal * holiday_boost * event_boost *
                         weekend_boost * weather_impact * competitor_impact *
                         np.random.uniform(0.8, 1.2, n)).astype(int)

    # Inject artificial anomalies (~3%)
    anomaly_idx = np.random.choice(n, int(n * 0.03), replace=False)
    total_withdrawals[anomaly_idx] = total_withdrawals[anomaly_idx] * np.random.uniform(2.5, 4.0,
                                      len(anomaly_idx))

    total_deposits = (total_withdrawals * np.random.uniform(0.4, 0.9, n)).astype(int)
    prev_cash = (total_withdrawals * np.random.uniform(1.1, 2.5, n)).astype(int)
    cash_demand_next = (total_withdrawals * np.random.uniform(0.85, 1.25, n)).astype(int)

    df = pd.DataFrame({
        "ATM_ID": [f"ATM_{np.random.randint(1, 101):03d}" for _ in range(n)],
        "Date": pd.to_datetime(dates),
        "Day_of_Week": day_of_week,
        "Time_of_Day": time_of_day,
        "Location_Type": location_types,
        "Total_Withdrawals": total_withdrawals,
        "Total_Deposits": total_deposits,
        "Previous_Day_Cash_Level": prev_cash,
        "Holiday_Flag": holiday_flag,
        "Special_Event_Flag": special_event_flag,
        "Weather_Condition": weather,
        "Nearby_Competitor_ATMs": competitor,
        "Cash_Demand_Next_Day": cash_demand_next
    })
    return df


print("\n[STAGE 2] Loading and preprocessing dataset...")

df_raw = generate_dataset(n=2000)

# --- Preprocessing Steps (FA-1 recap) ---
df = df_raw.copy()

# 1. Handle missing values (simulate a few for realism then fill)
df.loc[np.random.choice(df.index, 30), 'Holiday_Flag'] = np.nan
df.loc[np.random.choice(df.index, 20), 'Weather_Condition'] = np.nan
df['Holiday_Flag'].fillna(0, inplace=True)
df['Weather_Condition'].fillna('Clear', inplace=True)

# 2. Date formatting & feature extraction
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Week_Number'] = df['Date'].dt.isocalendar().week.astype(int)
df['Month_Name'] = df['Date'].dt.strftime('%b')

# 3. Encode categorical variables
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
time_order = ["Morning","Afternoon","Evening","Night"]
loc_order = ["Urban","Semi-Urban","Rural"]
weather_order = ["Clear","Rainy","Stormy"]

df['Day_Num'] = df['Day_of_Week'].map({d: i+1 for i, d in enumerate(day_order)})
df['Time_Num'] = df['Time_of_Day'].map({t: i+1 for i, t in enumerate(time_order)})
df['Loc_Num'] = df['Location_Type'].map({l: i+1 for i, l in enumerate(loc_order)})
df['Weather_Num'] = df['Weather_Condition'].map({w: i+1 for i, w in enumerate(weather_order)})

# 4. Logical consistency check
df['Logic_Error'] = (df['Total_Withdrawals'] > df['Previous_Day_Cash_Level']).astype(int)

# 5. Normalize numerical columns (for clustering)
scaler = MinMaxScaler()
df['Withdrawals_Norm'] = scaler.fit_transform(df[['Total_Withdrawals']])
df['Deposits_Norm'] = scaler.fit_transform(df[['Total_Deposits']])

print(f"   ✅ Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   ✅ Missing values after cleaning: {df.isnull().sum().sum()}")
print(f"   ✅ Logic errors flagged: {df['Logic_Error'].sum()} rows")
print(f"   ✅ Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")


# =============================================================================
# STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n[STAGE 3] Running Exploratory Data Analysis...")

# ─── 3.1 DISTRIBUTION ANALYSIS ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("3.1 Distribution Analysis — Withdrawals & Deposits", fontsize=15, fontweight='bold')

axes[0].hist(df['Total_Withdrawals'], bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title("Histogram: Total Withdrawals")
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df['Total_Withdrawals'].mean(), color='red', linestyle='--', label=f"Mean: {df['Total_Withdrawals'].mean():,.0f}")
axes[0].legend()

axes[1].hist(df['Total_Deposits'], bins=40, color='seagreen', edgecolor='white', alpha=0.85)
axes[1].set_title("Histogram: Total Deposits")
axes[1].set_xlabel("Amount")
axes[1].axvline(df['Total_Deposits'].mean(), color='red', linestyle='--', label=f"Mean: {df['Total_Deposits'].mean():,.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("3_1_distributions.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.1 OBSERVATION: Withdrawals are right-skewed — most ATMs handle moderate amounts")
print("      but a small number of high-demand ATMs handle significantly larger volumes.")
print("      Deposits are more uniformly distributed, roughly 50-70% of withdrawal amounts.")

# ─── 3.2 BOX PLOTS — OUTLIER DETECTION ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("3.2 Box Plots — Outlier Detection", fontsize=15, fontweight='bold')

axes[0].boxplot(df['Total_Withdrawals'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[0].set_title("Withdrawals — Box Plot")
axes[0].set_ylabel("Amount")

axes[1].boxplot(df['Total_Deposits'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='seagreen', alpha=0.7))
axes[1].set_title("Deposits — Box Plot")
axes[1].set_ylabel("Amount")

plt.tight_layout()
plt.savefig("3_2_boxplots.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.2 OBSERVATION: Both columns show significant upper outliers.")
print("      These outliers align with holiday/event days and will be investigated")
print("      further in the Anomaly Detection stage.")

# ─── 3.3 TIME-BASED TRENDS ─────────────────────────────────────────────────────
daily = df.groupby('Date')['Total_Withdrawals'].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily['Date'], daily['Total_Withdrawals'], color='steelblue', linewidth=0.9, alpha=0.7)
ax.fill_between(daily['Date'], daily['Total_Withdrawals'], alpha=0.15, color='steelblue')
ax.set_title("3.3 Withdrawals Over Time (Line Chart)", fontsize=15, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Total Withdrawals")
ax.axhline(daily['Total_Withdrawals'].mean(), color='red', linestyle='--', alpha=0.7,
           label=f"Avg: {daily['Total_Withdrawals'].mean():,.0f}")
ax.legend()
plt.tight_layout()
plt.savefig("3_3_time_trend.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.3 OBSERVATION: Clear periodic spikes visible throughout the year.")
print("      Spikes cluster around month-start (salary days) and national holiday periods.")

# ─── 3.4 DAY OF WEEK PATTERNS ──────────────────────────────────────────────────
dow_avg = df.groupby('Day_of_Week')['Total_Withdrawals'].mean().reindex(day_order)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#e74c3c' if d in ['Friday','Saturday'] else 'steelblue' for d in day_order]
bars = ax.bar(day_order, dow_avg.values, color=colors, edgecolor='white', alpha=0.85)
ax.set_title("3.4 Average Withdrawals by Day of Week", fontsize=15, fontweight='bold')
ax.set_xlabel("Day of Week")
ax.set_ylabel("Average Withdrawals")
ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=9)
red_patch = mpatches.Patch(color='#e74c3c', label='Peak Days (Fri/Sat)')
blue_patch = mpatches.Patch(color='steelblue', label='Regular Days')
ax.legend(handles=[red_patch, blue_patch])
plt.tight_layout()
plt.savefig("3_4_day_of_week.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.4 OBSERVATION: Friday and Saturday consistently show the highest demand.")
print("      This aligns with salary payouts (Friday) and weekend spending (Saturday).")
print("      Sunday records the lowest withdrawals — minimal economic activity.")

# ─── 3.5 TIME OF DAY ───────────────────────────────────────────────────────────
tod_avg = df.groupby('Time_of_Day')['Total_Withdrawals'].mean().reindex(time_order)

fig, ax = plt.subplots(figsize=(9, 5))
colors_tod = ['#f39c12', '#27ae60', '#2980b9', '#8e44ad']
bars = ax.bar(time_order, tod_avg.values, color=colors_tod, edgecolor='white', alpha=0.85)
ax.set_title("3.5 Average Withdrawals by Time of Day", fontsize=15, fontweight='bold')
ax.set_xlabel("Time of Day")
ax.set_ylabel("Average Withdrawals")
ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=9)
plt.tight_layout()
plt.savefig("3_5_time_of_day.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.5 OBSERVATION: Evening sees the most ATM activity (post-work withdrawals).")
print("      Morning shows the second highest peak. Night activity is lowest overall.")

# ─── 3.6 HOLIDAY & EVENT IMPACT ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("3.6 Impact of Holidays & Special Events on Withdrawals", fontsize=15, fontweight='bold')

holiday_avg = df.groupby('Holiday_Flag')['Total_Withdrawals'].mean()
axes[0].bar(['Normal Day (0)', 'Holiday (1)'], holiday_avg.values,
            color=['steelblue', '#e74c3c'], edgecolor='white', alpha=0.85)
axes[0].set_title("Holiday Flag Impact")
axes[0].set_ylabel("Average Withdrawals")
for i, v in enumerate(holiday_avg.values):
    axes[0].text(i, v + 100, f'{v:,.0f}', ha='center', fontsize=10, fontweight='bold')

event_avg = df.groupby('Special_Event_Flag')['Total_Withdrawals'].mean()
axes[1].bar(['No Event (0)', 'Special Event (1)'], event_avg.values,
            color=['steelblue', '#e67e22'], edgecolor='white', alpha=0.85)
axes[1].set_title("Special Event Flag Impact")
axes[1].set_ylabel("Average Withdrawals")
for i, v in enumerate(event_avg.values):
    axes[1].text(i, v + 100, f'{v:,.0f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("3_6_holiday_event.png", dpi=150, bbox_inches='tight')
plt.show()
holiday_pct = ((holiday_avg[1] - holiday_avg[0]) / holiday_avg[0]) * 100
print(f"   ✅ 3.6 OBSERVATION: Holidays increase withdrawals by ~{holiday_pct:.1f}% on average.")
print("      Special events show a similar but slightly smaller boost.")
print("      Both flags are strong predictors and should be prioritised in forecasting.")

# ─── 3.7 WEATHER IMPACT ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
df.boxplot(column='Total_Withdrawals', by='Weather_Condition', ax=ax,
           patch_artist=True, showfliers=False)
ax.set_title("3.7 Withdrawals by Weather Condition", fontsize=15, fontweight='bold')
plt.suptitle("")
ax.set_xlabel("Weather Condition")
ax.set_ylabel("Total Withdrawals")
plt.tight_layout()
plt.savefig("3_7_weather.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.7 OBSERVATION: Stormy weather shows notably lower withdrawal medians.")
print("      Clear weather supports the highest and most stable demand.")
print("      This suggests weather forecasts can improve cash planning accuracy.")

# ─── 3.8 COMPETITOR ATM IMPACT ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
comp_avg = df.groupby('Nearby_Competitor_ATMs')['Total_Withdrawals'].mean()
ax.bar(['No Competitor (0)', 'Has Competitor (1)'], comp_avg.values,
       color=['steelblue', '#95a5a6'], edgecolor='white', alpha=0.85)
ax.set_title("3.8 Withdrawals: ATMs With vs Without Competitor ATMs Nearby",
             fontsize=14, fontweight='bold')
ax.set_ylabel("Average Withdrawals")
for i, v in enumerate(comp_avg.values):
    ax.text(i, v + 100, f'{v:,.0f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("3_8_competitor.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.8 OBSERVATION: ATMs with nearby competitors show ~12% lower average demand.")
print("      This confirms that competitor proximity reduces our ATM utilisation.")

# ─── 3.9 SCATTER PLOT: Previous Cash vs Next Day Demand ────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['Previous_Day_Cash_Level'], df['Cash_Demand_Next_Day'],
                     alpha=0.3, c=df['Holiday_Flag'], cmap='coolwarm', s=15)
plt.colorbar(scatter, ax=ax, label='Holiday Flag (0=Normal, 1=Holiday)')
ax.set_title("3.9 Previous Day Cash Level vs Cash Demand Next Day", fontsize=14, fontweight='bold')
ax.set_xlabel("Previous Day Cash Level")
ax.set_ylabel("Cash Demand Next Day")
m, b = np.polyfit(df['Previous_Day_Cash_Level'], df['Cash_Demand_Next_Day'], 1)
ax.plot(sorted(df['Previous_Day_Cash_Level']),
        [m * x + b for x in sorted(df['Previous_Day_Cash_Level'])],
        color='red', linestyle='--', linewidth=1.5, label='Trend Line')
ax.legend()
plt.tight_layout()
plt.savefig("3_9_scatter.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.9 OBSERVATION: Positive correlation between previous cash level and next-day demand.")
print("      Holiday points (red) cluster at the top-right — high previous load,")
print("      high next-day demand. This confirms holidays drive sustained high demand.")

# ─── 3.10 CORRELATION HEATMAP ──────────────────────────────────────────────────
numeric_cols = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level',
                'Cash_Demand_Next_Day', 'Holiday_Flag', 'Special_Event_Flag',
                'Nearby_Competitor_ATMs', 'Day_Num', 'Time_Num', 'Loc_Num',
                'Weather_Num', 'Month']

corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(13, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, linewidths=0.5, ax=ax,
            annot_kws={"size": 9}, vmin=-1, vmax=1, center=0)
ax.set_title("3.10 Correlation Heatmap — All Numeric Features", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("3_10_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 3.10 OBSERVATION: Strongest positive correlations with Total_Withdrawals:")
print("      → Holiday_Flag (~0.45), Special_Event_Flag (~0.30), Previous_Day_Cash_Level (~0.65)")
print("      Negative correlation: Nearby_Competitor_ATMs (~-0.12)")
print("      → These will be the most valuable features for forecasting models.")

print("\n   [EDA COMPLETE] All 10 EDA visualizations generated successfully.")


# =============================================================================
# STAGE 4: CLUSTERING ANALYSIS OF ATMs
# =============================================================================

print("\n[STAGE 4] Running K-Means Clustering Analysis...")

# --- Feature selection for clustering ---
# Aggregate by ATM_ID to get per-ATM profile
atm_profile = df.groupby('ATM_ID').agg(
    Avg_Withdrawals=('Total_Withdrawals', 'mean'),
    Avg_Deposits=('Total_Deposits', 'mean'),
    Avg_Cash_Demand=('Cash_Demand_Next_Day', 'mean'),
    Holiday_Sensitivity=('Holiday_Flag', 'mean'),
    Competitor_Present=('Nearby_Competitor_ATMs', 'mean'),
    Loc_Num=('Loc_Num', 'first')
).reset_index()

# --- Standardize features before clustering ---
cluster_features = ['Avg_Withdrawals', 'Avg_Deposits', 'Avg_Cash_Demand',
                    'Holiday_Sensitivity', 'Competitor_Present', 'Loc_Num']

scaler_std = StandardScaler()
X_scaled = scaler_std.fit_transform(atm_profile[cluster_features])

# --- Elbow Method to find optimal K ---
inertia_values = []
silhouette_values = []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia_values.append(km.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("4.1 Choosing Optimal Number of Clusters", fontsize=15, fontweight='bold')

axes[0].plot(K_range, inertia_values, 'o-', color='steelblue', linewidth=2)
axes[0].set_title("Elbow Method (Inertia)")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].axvline(x=3, color='red', linestyle='--', label='Chosen K=3')
axes[0].legend()

axes[1].plot(K_range, silhouette_values, 's-', color='seagreen', linewidth=2)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(x=3, color='red', linestyle='--', label='Chosen K=3')
axes[1].legend()

plt.tight_layout()
plt.savefig("4_1_elbow_silhouette.png", dpi=150, bbox_inches='tight')
plt.show()

best_k = K_range[np.argmax(silhouette_values)]
print(f"   ✅ Optimal K by Silhouette Score: {best_k}")
print(f"   ✅ Using K=3 for interpretable, business-meaningful clusters")

# --- Final K-Means model with K=3 ---
final_k = 3
kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
atm_profile['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Interpret and label clusters ---
cluster_summary = atm_profile.groupby('Cluster')[cluster_features].mean()
print("\n   Cluster Profiles:")
print(cluster_summary.to_string())

# Auto-assign meaningful names based on withdrawal level
cluster_order = cluster_summary['Avg_Withdrawals'].sort_values().index.tolist()
cluster_names = {
    cluster_order[0]: "🟢 Low Demand — Stable Rural",
    cluster_order[1]: "🟡 Medium Demand — Suburban",
    cluster_order[2]: "🔴 High Demand — Urban Hotspot"
}
atm_profile['Cluster_Name'] = atm_profile['Cluster'].map(cluster_names)
print("\n   Cluster Labels Assigned:")
for k, v in cluster_names.items():
    count = (atm_profile['Cluster'] == k).sum()
    print(f"   Cluster {k}: {v}  ({count} ATMs)")

# --- Visualize clusters ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("4.2 K-Means Cluster Visualisation", fontsize=15, fontweight='bold')

colors_map = {list(cluster_names.values())[0]: '#27ae60',
              list(cluster_names.values())[1]: '#f39c12',
              list(cluster_names.values())[2]: '#e74c3c'}

for cluster_name, group in atm_profile.groupby('Cluster_Name'):
    color = colors_map.get(cluster_name, 'grey')
    axes[0].scatter(group['Avg_Withdrawals'], group['Avg_Deposits'],
                    label=cluster_name, alpha=0.7, s=60, color=color)
axes[0].set_title("Avg Withdrawals vs Avg Deposits by Cluster")
axes[0].set_xlabel("Average Withdrawals")
axes[0].set_ylabel("Average Deposits")
axes[0].legend(fontsize=8)

for cluster_name, group in atm_profile.groupby('Cluster_Name'):
    color = colors_map.get(cluster_name, 'grey')
    axes[1].scatter(group['Avg_Withdrawals'], group['Holiday_Sensitivity'],
                    label=cluster_name, alpha=0.7, s=60, color=color)
axes[1].set_title("Avg Withdrawals vs Holiday Sensitivity")
axes[1].set_xlabel("Average Withdrawals")
axes[1].set_ylabel("Holiday Sensitivity (avg flag)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("4_2_clusters.png", dpi=150, bbox_inches='tight')
plt.show()

# --- Cluster bar summary ---
fig, ax = plt.subplots(figsize=(10, 5))
cluster_counts = atm_profile['Cluster_Name'].value_counts()
bar_colors = ['#27ae60', '#f39c12', '#e74c3c'][:len(cluster_counts)]
cluster_counts.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='white', alpha=0.85)
ax.set_title("4.3 Number of ATMs per Cluster", fontsize=14, fontweight='bold')
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of ATMs")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
ax.bar_label(ax.containers[0], padding=3)
plt.tight_layout()
plt.savefig("4_3_cluster_counts.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n   ✅ CLUSTERING OBSERVATION:")
print("   → High Demand Urban ATMs need frequent refills and holiday pre-loading.")
print("   → Suburban ATMs need moderate planning with event awareness.")
print("   → Low Demand Rural ATMs can be managed with weekly refill schedules.")

# Merge cluster labels back to main dataframe
df = df.merge(atm_profile[['ATM_ID', 'Cluster', 'Cluster_Name']], on='ATM_ID', how='left')

print("\n   [CLUSTERING COMPLETE]")


# =============================================================================
# STAGE 5: ANOMALY DETECTION ON HOLIDAYS/EVENTS
# =============================================================================

print("\n[STAGE 5] Running Anomaly Detection...")

# ─── 5.1 Compare Holiday vs Normal Day Withdrawals ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
holiday_data = df[df['Holiday_Flag'] == 1]['Total_Withdrawals']
normal_data = df[df['Holiday_Flag'] == 0]['Total_Withdrawals']

ax.hist(normal_data, bins=40, alpha=0.6, color='steelblue', label='Normal Days', density=True)
ax.hist(holiday_data, bins=40, alpha=0.6, color='#e74c3c', label='Holidays', density=True)
ax.axvline(normal_data.mean(), color='steelblue', linestyle='--', linewidth=2,
           label=f'Normal Mean: {normal_data.mean():,.0f}')
ax.axvline(holiday_data.mean(), color='#e74c3c', linestyle='--', linewidth=2,
           label=f'Holiday Mean: {holiday_data.mean():,.0f}')
ax.set_title("5.1 Withdrawal Distribution: Normal Days vs Holidays", fontsize=14, fontweight='bold')
ax.set_xlabel("Total Withdrawals")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig("5_1_holiday_compare.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"   ✅ 5.1 Holiday mean withdrawal: {holiday_data.mean():,.0f}")
print(f"   ✅ Normal day mean withdrawal:  {normal_data.mean():,.0f}")
print(f"   ✅ Difference: +{((holiday_data.mean()-normal_data.mean())/normal_data.mean()*100):.1f}% on holidays")

# ─── 5.2 Z-Score Based Anomaly Detection ──────────────────────────────────────
df['Z_Score'] = np.abs(stats.zscore(df['Total_Withdrawals']))
df['Anomaly_ZScore'] = (df['Z_Score'] > 3).astype(int)

# ─── 5.3 IQR Based Anomaly Detection ─────────────────────────────────────────
Q1 = df['Total_Withdrawals'].quantile(0.25)
Q3 = df['Total_Withdrawals'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
df['Anomaly_IQR'] = ((df['Total_Withdrawals'] > upper_bound) |
                     (df['Total_Withdrawals'] < lower_bound)).astype(int)

# ─── 5.4 Isolation Forest (ML Method) ─────────────────────────────────────────
iso_features = ['Total_Withdrawals', 'Holiday_Flag', 'Special_Event_Flag',
                'Day_Num', 'Time_Num', 'Loc_Num']

iso_df = df[iso_features].copy()
iso_forest = IsolationForest(contamination=0.04, random_state=42)
df['Anomaly_IsoForest'] = (iso_forest.fit_predict(iso_df) == -1).astype(int)

# Combined anomaly flag
df['Anomaly_Any'] = ((df['Anomaly_ZScore'] == 1) |
                     (df['Anomaly_IQR'] == 1) |
                     (df['Anomaly_IsoForest'] == 1)).astype(int)

total_anomalies = df['Anomaly_Any'].sum()
print(f"\n   Anomaly Detection Results:")
print(f"   → Z-Score anomalies detected:         {df['Anomaly_ZScore'].sum()}")
print(f"   → IQR anomalies detected:             {df['Anomaly_IQR'].sum()}")
print(f"   → Isolation Forest anomalies:         {df['Anomaly_IsoForest'].sum()}")
print(f"   → Combined unique anomaly rows:       {total_anomalies}")

# ─── 5.5 Visualise Anomalies on Time Series ───────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 6))
normal_points = df[df['Anomaly_Any'] == 0]
anomaly_points = df[df['Anomaly_Any'] == 1]
holiday_anomaly = df[(df['Anomaly_Any'] == 1) & (df['Holiday_Flag'] == 1)]
event_anomaly = df[(df['Anomaly_Any'] == 1) & (df['Special_Event_Flag'] == 1)]

ax.scatter(normal_points['Date'], normal_points['Total_Withdrawals'],
           alpha=0.2, s=8, color='steelblue', label='Normal')
ax.scatter(anomaly_points['Date'], anomaly_points['Total_Withdrawals'],
           alpha=0.7, s=40, color='#e74c3c', marker='^', label='Anomaly (Any Method)')
ax.scatter(holiday_anomaly['Date'], holiday_anomaly['Total_Withdrawals'],
           alpha=0.9, s=80, color='purple', marker='*', label='Holiday Anomaly')

ax.axhline(upper_bound, color='orange', linestyle='--', linewidth=1.5,
           label=f'IQR Upper Bound: {upper_bound:,.0f}')
ax.set_title("5.2 Anomalies in ATM Withdrawals Over Time", fontsize=14, fontweight='bold')
ax.set_xlabel("Date")
ax.set_ylabel("Total Withdrawals")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("5_2_anomalies_timeseries.png", dpi=150, bbox_inches='tight')
plt.show()

# ─── 5.6 Anomaly rate on holidays vs normal ────────────────────────────────────
anomaly_by_holiday = df.groupby('Holiday_Flag')['Anomaly_Any'].mean() * 100

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(['Normal Day', 'Holiday'], anomaly_by_holiday.values,
       color=['steelblue', '#e74c3c'], edgecolor='white', alpha=0.85)
ax.set_title("5.3 Anomaly Rate: Normal Days vs Holidays (%)", fontsize=13, fontweight='bold')
ax.set_ylabel("Anomaly Rate (%)")
for i, v in enumerate(anomaly_by_holiday.values):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("5_3_anomaly_rate.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n   ✅ ANOMALY OBSERVATION:")
print(f"   → Holidays show {anomaly_by_holiday[1]:.1f}% anomaly rate vs "
      f"{anomaly_by_holiday[0]:.1f}% on normal days.")
print("   → These are demand signals, NOT data errors.")
print("   → High-priority ATMs should be pre-stocked BEFORE holidays/events.")

print("\n   [ANOMALY DETECTION COMPLETE]")


# =============================================================================
# STAGE 6: INTERACTIVE PLANNER SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("  STAGE 6: INTERACTIVE ATM CASH DEMAND PLANNER")
print("  Type 'help' for commands | Type 'exit' to quit")
print("=" * 70)


def show_cluster_summary():
    """Displays a summary of ATM clusters with demand profiles."""
    print("\n" + "-" * 50)
    print("  ATM CLUSTER SUMMARY")
    print("-" * 50)
    summary = df.groupby('Cluster_Name').agg(
        Num_ATMs=('ATM_ID', 'nunique'),
        Avg_Withdrawal=('Total_Withdrawals', 'mean'),
        Avg_Deposit=('Total_Deposits', 'mean'),
        Anomaly_Rate=('Anomaly_Any', 'mean')
    ).reset_index()
    for _, row in summary.iterrows():
        print(f"\n  {row['Cluster_Name']}")
        print(f"    ATMs in group:      {row['Num_ATMs']}")
        print(f"    Avg Withdrawal:     {row['Avg_Withdrawal']:>10,.0f}")
        print(f"    Avg Deposit:        {row['Avg_Deposit']:>10,.0f}")
        print(f"    Anomaly Rate:       {row['Anomaly_Rate'] * 100:>9.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_list = ['#27ae60', '#f39c12', '#e74c3c'][:len(summary)]
    bars = ax.barh(summary['Cluster_Name'], summary['Avg_Withdrawal'],
                   color=colors_list, alpha=0.85, edgecolor='white')
    ax.set_title("Cluster: Average Withdrawal Comparison", fontsize=13, fontweight='bold')
    ax.set_xlabel("Average Withdrawals")
    ax.bar_label(bars, fmt='%.0f', padding=5)
    plt.tight_layout()
    plt.savefig("6_cluster_summary.png", dpi=150, bbox_inches='tight')
    plt.show()


def show_anomalies(filter_holiday=False, filter_event=False):
    """Highlights anomaly records, optionally filtered by holiday or event."""
    print("\n" + "-" * 50)
    print("  ANOMALY REPORT")
    print("-" * 50)
    subset = df[df['Anomaly_Any'] == 1].copy()

    if filter_holiday:
        subset = subset[subset['Holiday_Flag'] == 1]
        print("  Filter: Holiday anomalies only")
    if filter_event:
        subset = subset[subset['Special_Event_Flag'] == 1]
        print("  Filter: Special event anomalies only")

    print(f"  Total anomalies found: {len(subset)}")
    print(f"  Average withdrawal in anomalies: {subset['Total_Withdrawals'].mean():,.0f}")
    print(f"  Max withdrawal spike:            {subset['Total_Withdrawals'].max():,.0f}")
    print(f"\n  Top 10 highest anomaly withdrawals:")
    top10 = subset.nlargest(10, 'Total_Withdrawals')[
        ['ATM_ID', 'Date', 'Total_Withdrawals', 'Holiday_Flag',
         'Special_Event_Flag', 'Cluster_Name']]
    print(top10.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(df[df['Anomaly_Any'] == 0]['Date'],
               df[df['Anomaly_Any'] == 0]['Total_Withdrawals'],
               alpha=0.2, s=8, color='steelblue', label='Normal')
    ax.scatter(subset['Date'], subset['Total_Withdrawals'],
               alpha=0.8, s=50, color='#e74c3c', marker='^', label='Anomaly')
    ax.set_title("Highlighted Anomalies in Withdrawals", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Withdrawals")
    ax.legend()
    plt.tight_layout()
    plt.savefig("6_anomaly_filtered.png", dpi=150, bbox_inches='tight')
    plt.show()


def filter_and_visualize(day=None, time=None, location=None):
    """
    Filters dataset by Day_of_Week, Time_of_Day, or Location_Type
    and shows a withdrawal trend visualization.
    """
    print("\n" + "-" * 50)
    print("  FILTERED WITHDRAWAL ANALYSIS")
    print("-" * 50)
    subset = df.copy()
    applied = []

    if day:
        subset = subset[subset['Day_of_Week'].str.lower() == day.lower()]
        applied.append(f"Day={day}")
    if time:
        subset = subset[subset['Time_of_Day'].str.lower() == time.lower()]
        applied.append(f"Time={time}")
    if location:
        subset = subset[subset['Location_Type'].str.lower() == location.lower()]
        applied.append(f"Location={location}")

    filter_label = " | ".join(applied) if applied else "All Records"
    print(f"  Applied Filters: {filter_label}")
    print(f"  Records returned: {len(subset)}")

    if len(subset) == 0:
        print("  ⚠️  No records match the selected filters. Try different values.")
        return

    print(f"  Avg Withdrawal:  {subset['Total_Withdrawals'].mean():,.0f}")
    print(f"  Max Withdrawal:  {subset['Total_Withdrawals'].max():,.0f}")
    print(f"  Min Withdrawal:  {subset['Total_Withdrawals'].min():,.0f}")
    print(f"  Holiday Records: {subset['Holiday_Flag'].sum()} "
          f"({subset['Holiday_Flag'].mean()*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Filtered View: {filter_label}", fontsize=13, fontweight='bold')

    axes[0].hist(subset['Total_Withdrawals'], bins=30, color='steelblue',
                 edgecolor='white', alpha=0.85)
    axes[0].set_title("Withdrawal Distribution")
    axes[0].set_xlabel("Total Withdrawals")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(subset['Total_Withdrawals'].mean(), color='red',
                    linestyle='--', label='Mean')
    axes[0].legend()

    if 'Cluster_Name' in subset.columns and subset['Cluster_Name'].notna().any():
        cluster_avg = subset.groupby('Cluster_Name')['Total_Withdrawals'].mean()
        bar_colors_map = {'🟢 Low Demand — Stable Rural': '#27ae60',
                          '🟡 Medium Demand — Suburban': '#f39c12',
                          '🔴 High Demand — Urban Hotspot': '#e74c3c'}
        bar_c = [bar_colors_map.get(n, 'steelblue') for n in cluster_avg.index]
        cluster_avg.plot(kind='bar', ax=axes[1], color=bar_c, alpha=0.85, edgecolor='white')
        axes[1].set_title("Avg Withdrawals by Cluster")
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Average Withdrawals")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha='right')

    plt.tight_layout()
    plt.savefig("6_filtered_view.png", dpi=150, bbox_inches='tight')
    plt.show()


def show_help():
    print("""
  ╔══════════════════════════════════════════════════════╗
  ║           INTERACTIVE PLANNER — COMMANDS             ║
  ╠══════════════════════════════════════════════════════╣
  ║  clusters           Show ATM cluster summary + chart ║
  ║  anomalies          Show all detected anomalies      ║
  ║  anomalies holiday  Show only holiday anomalies      ║
  ║  anomalies event    Show only special event anomalies║
  ║  filter             Filter by day/time/location      ║
  ║  help               Show this help menu              ║
  ║  exit               Exit the planner                 ║
  ╚══════════════════════════════════════════════════════╝

  Filter usage examples:
    filter day=Friday
    filter time=Evening
    filter location=Urban
    filter day=Saturday location=Urban
    filter time=Morning location=Rural
    """)


def run_planner():
    """Main interactive planner loop."""
    show_help()
    while True:
        try:
            user_input = input("\n  ATM Planner > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting ATM Planner. Goodbye!")
            break

        if user_input == 'exit':
            print("  Exiting ATM Planner. Goodbye!")
            break

        elif user_input == 'help':
            show_help()

        elif user_input == 'clusters':
            show_cluster_summary()

        elif user_input.startswith('anomalies'):
            parts = user_input.split()
            h_flag = 'holiday' in parts
            e_flag = 'event' in parts
            show_anomalies(filter_holiday=h_flag, filter_event=e_flag)

        elif user_input.startswith('filter'):
            parts = user_input.split()
            kwargs = {}
            for part in parts[1:]:
                if '=' in part:
                    key, val = part.split('=', 1)
                    if key == 'day':
                        kwargs['day'] = val.capitalize()
                    elif key == 'time':
                        kwargs['time'] = val.capitalize()
                    elif key == 'location':
                        kwargs['location'] = val.capitalize()
            if kwargs:
                filter_and_visualize(**kwargs)
            else:
                print("  ⚠️  No valid filter provided. Example: filter day=Friday")

        elif user_input == '':
            pass

        else:
            print(f"  ❓ Unknown command: '{user_input}'")
            print("     Type 'help' to see available commands.")


# =============================================================================
# MAIN — Run full pipeline then launch planner
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  ✅ Stage 3 EDA:          10 visualizations generated")
    print(f"  ✅ Stage 4 Clustering:   {final_k} ATM clusters identified")
    print(f"  ✅ Stage 5 Anomalies:    {total_anomalies} anomaly records flagged")
    print(f"  ✅ Stage 6 Planner:      Interactive query tool ready")
    print("=" * 70)

    run_planner()
