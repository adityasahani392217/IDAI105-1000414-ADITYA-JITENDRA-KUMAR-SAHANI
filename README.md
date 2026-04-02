# 🏧 ATM Demand Forecasting & Intelligence Dashboard

**CRS: Artificial Intelligence &nbsp;|&nbsp; Course: Data Mining &nbsp;|&nbsp; Formative Assessment 2 &nbsp;|&nbsp; 60 Marks**

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.14%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

> **An end-to-end interactive data mining dashboard for FinTrust Bank Ltd. — transforming raw ATM transaction records into cash management intelligence through EDA, K-Means clustering, anomaly detection, and a real-time interactive planner.**

🌐 **Live App:** [Launch on Streamlit Cloud](https://idai105-1000414-aditya-jitendra-kumar-sahani.streamlit.app/)

---

## 📑 Table of Contents

1. [Project Scope](#-project-scope)
2. [Repository Structure](#-repository-structure)
3. [Dataset](#-dataset)
4. [Preprocessing](#-preprocessing)
5. [EDA Visualizations](#-eda-visualizations)
6. [Clustering Analysis](#-clustering-analysis)
7. [Anomaly Detection](#-anomaly-detection)
8. [Interactive Cash Demand Planner](#-interactive-cash-demand-planner)
9. [Strategic Recommendations](#-strategic-recommendations)
10. [Visual Gallery & Full Walkthrough](#-visual-gallery--full-walkthrough)
11. [Installation & Local Run](#-installation--local-run)
12. [Deployment](#-deployment)
13. [Troubleshooting](#-troubleshooting)
14. [Learning Outcomes](#-learning-outcomes)
15. [References](#-references)
16. [Submission Details](#-submission-details)

---

## 🎯 Project Scope

FinTrust Bank Ltd. operates a network of **50 ATMs** across multiple location types. Inefficient cash management — either overstocking (idle capital) or understocking (customer frustration) — costs the bank significantly. This dashboard solves that by mining two years of historical transaction patterns and surfacing data-driven operational decisions.

**Core objectives:**

- Perform **Exploratory Data Analysis** across 5,658 transaction records — distributions, temporal patterns, holiday/event impact, and feature correlations
- **Cluster ATMs into 3 behavioural segments** using K-Means with Elbow Method + Silhouette Score validation
- **Detect withdrawal anomalies** using two complementary methods: IQR statistical thresholds and Isolation Forest (ML)
- Deliver a **real-time Interactive Cash Demand Planner** with sidebar filters and actionable recommendations
- Host the complete solution as a **public Streamlit web app**

---

## 📁 Repository Structure

```
atm-demand-forecasting/
│
├── app.py                           # 🚀 Main Streamlit application (500+ lines)
├── requirements.txt                 # 📦 Python dependencies with pinned versions
├── atm_cash_management_dataset.csv  # 📊 ATM transaction dataset (5,658 records)
├── README.md                        # 📖 This file
│
├── .streamlit/
│   └── config.toml                  # 🎨 Theme (primaryColor #FF4B4B) & server config
│
└── Screenshot/                      # 🖼️ All dashboard screenshots (16 files)
    ├── Actionable Recommendations.png
    ├── Anomaly Detection on Withdrawal.png
    ├── Anomaly Map  Withdrawals by Date & Clus....png
    ├── Anomaly Rate by Location Type.png
    ├── Anomaly Rates by Holiday & Event.png
    ├── Cluster Profiles.png
    ├── Cluster Visualization (PCA Projection).png
    ├── Dashboard.png
    ├── Detailed Transaction Table.png
    ├── Distribution analysis.png
    ├── Holiday Event Impact & External Factors.png
    ├── Interactive Cash Demand Planner.png
    ├── Optimal Cluster Selection.png
    ├── Relationship Analysis.png
    ├── Time Based Trend.png          ← day-of-week & time-of-day bar charts
    └── Time Based Trends.png         ← full daily time series (2022–2024)
```

> **Note on duplicate screenshot names:** `Time Based Trend.png` and `Time Based Trends.png` are two distinct screenshots — the former shows the day-of-week and time-of-day breakdown charts; the latter shows the full aggregated daily line chart.

---

## 📊 Dataset

**Source:** FinTrust Bank Ltd. (provided via course portal)
**Records:** 5,658 transactions &nbsp;|&nbsp; **ATMs:** 50 unique &nbsp;|&nbsp; **Date Range:** January 2022 – January 2024

| Column | Data Type | Description | Example |
|--------|-----------|-------------|---------|
| `ATM_ID` | String | Unique ATM identifier | `ATM_0041` |
| `Date` | Date | Transaction date | `2022-04-25` |
| `Day_of_Week` | String | Monday through Sunday | `Monday` |
| `Time_of_Day` | String | Morning / Afternoon / Evening / Night | `Morning` |
| `Total_Withdrawals` | Integer | Cash withdrawn (₹) | `57,450` |
| `Total_Deposits` | Integer | Cash deposited (₹) | `9,308` |
| `Location_Type` | String | Bank Branch / Mall / Supermarket / Gas Station / Standalone | `Standalone` |
| `Holiday_Flag` | Binary | `1` = public holiday, `0` = normal day | `0` |
| `Special_Event_Flag` | Binary | `1` = special event nearby, `0` = none | `1` |
| `Previous_Day_Cash_Level` | Integer | Opening cash inventory for the day (₹) | `112,953` |
| `Weather_Condition` | String | Clear / Rainy / Snowy / Cloudy | `Rainy` |
| `Nearby_Competitor_ATMs` | Integer | Competing ATMs within the vicinity | `5` |
| `Cash_Demand_Next_Day` | Integer | Target variable — forecasted next-day demand (₹) | `44,165` |

---

## 🔧 Preprocessing

All preprocessing is handled inside `app.py` under `load_data()`, decorated with `@st.cache_data` so it executes only once per session regardless of how many widgets the user interacts with.

| Step | Method | Detail |
|------|--------|--------|
| **Date Parsing** | `pd.to_datetime()` | Converts the `Date` column; extracts `Month` and `Year` as integer columns |
| **Weekend Flag** | Boolean mask | `Is_Weekend = Day_of_Week.isin(['Saturday','Sunday']).astype(int)` |
| **Label Encoding** | `sklearn.LabelEncoder` | Applied to `Location_Type` to produce a numeric feature for K-Means input |
| **Feature Scaling** | `sklearn.StandardScaler` | Normalises withdrawal, deposit, cash level, and competitor count before clustering |
| **PCA Reduction** | `sklearn.PCA(n_components=2)` | Compresses five clustering features to 2D for the visual scatter projection |
| **IQR Thresholds** | Q1 − 1.5×IQR / Q3 + 1.5×IQR | Dynamic lower and upper bounds computed on the full (unfiltered) dataset |
| **Isolation Forest** | `contamination=0.05` | ML-based anomaly scorer available as a toggle alternative to IQR |

---

## 📈 EDA Visualizations

**Stage 3** covers five sub-sections with 10+ interactive Plotly charts. Every chart ends with a blue observation callout explaining the business implication.

### 3-B · Distribution Analysis

Histograms with overlaid box plots for Total Withdrawals and Total Deposits.

| Finding | Detail |
|---------|--------|
| Withdrawals are **right-skewed** | Most days cluster between ₹35k–60k; a long right tail represents holidays and paydays reaching ₹100k+ |
| Deposits are **lower and similarly skewed** | Consistently lower than withdrawals — confirms ATMs are net cash dispensers |
| Box plots highlight **upper outliers** | Withdrawal outliers reach ₹100k+; deposit outliers reach ₹30k+ |

### 3-C & 3-D · Time-Based Trends

Daily time series, day-of-week bar chart, and time-of-day bar chart.

| Finding | Detail |
|---------|--------|
| **Periodic weekly spikes** visible | Spikes align with weekends and public holiday clusters across the full 2022–2024 window |
| **Monday is lowest** | Post-weekend cash depletion — customers withdrew over the weekend |
| **Friday is elevated** | Classic payday effect — end-of-week spending surge |
| **Weekends are highest overall** | Saturday and Sunday consistently the tallest bars in the day-of-week chart |
| **Morning is peak time-of-day** | Highest demand in the first session; Afternoon is the lowest |
| **Evening rebounds** | After-work and shopping hours drive a secondary demand peak |

### 3-E · Holiday & Event Impact and External Factors

Holiday vs normal day comparison, special event vs no-event comparison, weather boxplots, and competitor ATM bar chart.

| Finding | Detail |
|---------|--------|
| **Holidays drive significantly higher withdrawals** | Clear step-up vs normal days — festive spending is the dominant driver |
| **Special events also elevate demand** | Concerts, sports events, festivals — though gap is smaller than holidays |
| **Weather has limited impact** | IQR ranges overlap across all four conditions — not an operational planning variable |
| **Competitor count is irrelevant** | No meaningful trend between 0–5 nearby ATMs and withdrawal volume |

### 3-F · Relationship Analysis

Scatter plot of cash levels vs next-day demand, plus full numeric correlation heatmap.

| Feature Pair | Correlation | Implication |
|-------------|-------------|-------------|
| Total Withdrawals ↔ Cash Demand Next Day | **r = 0.895** | Strongest signal — today's withdrawals predict tomorrow's demand with high reliability |
| Total Deposits ↔ Cash Demand Next Day | r = −0.214 | Moderate negative — higher deposits slightly suppress next-day demand |
| Previous Cash Level ↔ Cash Demand Next Day | r ≈ −0.006 | Near zero — leftover cash is not a useful predictor |
| Nearby Competitor ATMs ↔ any feature | r ≈ 0.00 | No meaningful relationship with any numeric variable |

---

## 🧠 Clustering Analysis

**Algorithm:** K-Means &nbsp;|&nbsp; **Optimal K:** 3 &nbsp;|&nbsp; **Validation:** Elbow Method + Silhouette Score

### How K = 3 Was Selected

The **Elbow Method** plots inertia against k = 2–10. The curve drops sharply from k=2 (~18k inertia) to k=3 (~16k) and then flattens — the inflection is marked with a red dashed line at k=3. The **Silhouette Score** peaks at k=2 (~0.194) but k=3 is preferred because three segments map cleanly to distinct ATM operational contexts (retail, captive-audience, and high-competition sites), offering actionable differentiation that two clusters cannot provide.

**Features used:** `Total_Withdrawals`, `Total_Deposits`, `Previous_Day_Cash_Level`, `Nearby_Competitor_ATMs`, `Location_Type` (label-encoded) — all scaled with StandardScaler before fitting.

### Cluster Profiles

| Cluster | Top Location | Avg Withdrawals | Avg Deposits | Avg Competitors | Share of Records | Operational Label |
|---------|-------------|----------------|-------------|----------------|-----------------|------------------|
| **0** | Supermarket | ₹49,918 | ₹10,179 | 2.7 | 41.7% | Retail foot-traffic ATMs |
| **1** | Mall | ₹49,781 | ₹9,974 | 0.9 | 25.5% | Captive-audience ATMs |
| **2** | Gas Station | ₹49,703 | ₹10,245 | 4.0 | 32.8% | High-competition transit ATMs |

**Interpretation guide:**
- **High-demand cluster (0)** → increase refill frequency, especially before weekends/holidays
- **Medium-demand cluster (1)** → standard schedule with holiday-sensitive adjustments
- **Low-demand cluster (2)** → reduce refill frequency; investigate if consistently low

The **PCA 2D projection** confirms meaningful spatial separation — each cluster forms a distinct region in the two-component space, validating that K-Means found genuine structure rather than noise.

---

## 🚨 Anomaly Detection

**Stage 5** offers two selectable detection methods via radio button: **IQR (Interquartile Range)** and **Isolation Forest (ML)**.

### IQR Results (Default)

| Metric | Value |
|--------|-------|
| Lower Bound | ₹8,474 |
| Upper Bound | ₹91,127 |
| **Total Anomalies Detected** | **33 records (~0.6% of 5,658)** |

### Anomaly Breakdown by Context

| Dimension | Normal Rate | Anomaly Rate | Key Insight |
|-----------|------------|-------------|-------------|
| Non-Holiday | 99.5% | **0.5%** | Baseline — investigate each for equipment fault or fraud |
| **Holiday** | 97.6% | **2.4%** | 4.8× higher than baseline — expected demand spikes; plan proactively |
| No Special Event | 99.4% | **0.6%** | Baseline event rate |
| Special Event | 99.5% | **0.5%** | Slightly lower than baseline — events are partially anticipated |

### Anomaly Rate by Location Type

| Location Type | Anomaly Rate | Recommended Action |
|---------------|-------------|-------------------|
| **Bank Branch** | **1.1%** | Tightest monitoring — highest risk |
| Supermarket | 0.7% | Review refill schedules; event-day pre-loading |
| Standalone | 0.5% | Standard monitoring |
| Mall | 0.3% | Low — captive audience behaviour is predictable |
| Gas Station | 0.3% | Low — consistent transit demand patterns |

---

## ⚙️ Interactive Cash Demand Planner

**Stage 6** integrates all pipeline outputs — EDA insights, cluster assignments, and anomaly flags — into a single real-time planning workspace.

### Sidebar Filters

| Filter | Type | Options |
|--------|------|---------|
| Day of Week | Multi-select | Monday / Tuesday / Wednesday / Thursday / Friday / Saturday / Sunday |
| Time of Day | Multi-select | Morning / Afternoon / Evening / Night |
| Location Type | Multi-select | Bank Branch / Gas Station / Mall / Standalone / Supermarket |
| Include Holidays | Checkbox | On / Off |
| Include Special Events | Checkbox | On / Off |

All charts, KPI cards, the transaction table, and the recommendations panel **update in real time** as filters change. A record counter in the sidebar (`Filtered records: X / 5,658`) gives immediate feedback on the active filter window.

### KPI Metrics (Unfiltered)

| Metric | Value |
|--------|-------|
| Total Records | 5,658 |
| Unique ATMs | 50 |
| Anomalies Detected | 33 |
| Anomaly Rate | 0.6% |

### Cluster Distribution (Unfiltered)

| Cluster | Top Location | Share |
|---------|-------------|-------|
| Cluster 0 | Supermarket | 41.7% |
| Cluster 2 | Gas Station | 32.8% |
| Cluster 1 | Mall | 25.5% |

---

## 🎓 Strategic Recommendations

Generated dynamically by the planner based on the current filter window:

| # | Recommendation | Evidence |
|---|---------------|----------|
| 1 | 🏦 **Prioritise high-demand cluster ATMs for more frequent refills** | Cluster 0 (Supermarket) handles 41.7% of all transactions |
| 2 | 🎉 **Pre-load ATMs at least one day before public holidays** | Holiday anomaly rate is 4.8× baseline (2.4% vs 0.5%) |
| 3 | 🎸 **Coordinate with event organisers to pre-load nearby ATMs** | Special events drive surges that can be anticipated with notice |
| 4 | ☀️ **Plan for higher-than-average withdrawals on clear weather weekends** | Clear + weekend is the highest-demand combination |
| 5 | 🔍 **Investigate non-holiday anomalies for equipment faults or fraud** | Non-holiday anomalies have no demand justification — each requires audit |
| 6 | 🏪 **Apply tightest monitoring to Bank Branch ATMs** | Highest anomaly rate at 1.1% across all location types |

---

## 📸 Visual Gallery & Full Walkthrough

Every screenshot below corresponds to a specific section of the deployed app, presented in the order a user encounters them. Each image is followed by a plain-English explanation of what is shown and what it means operationally.

---

### 🏠 Stage 3-A — Main Dashboard

> **What you see:** The full app entry point. The left sidebar shows interactive filters — Day of Week, Time of Day, Location Type — with multi-select tags and two checkboxes (Include Holidays, Include Special Events). A record counter confirms `Filtered records: 5,658 / 5,658` when nothing is filtered. The main panel shows the app title (`ATM Demand Forecasting & Insights`), a green dataset confirmation banner, and four navigation tabs (Exploratory Data Analysis / Clustering ATMs / Anomaly Detection / Interactive Planner).
>
> **Why it matters:** The sidebar-driven filtering architecture means every analytical view in the app is contextual — operations staff can zero in on, say, Friday evenings at Supermarkets near a sports venue and get instantaneous tailored insights.

![Dashboard](Screenshot/Dashboard.png)

---

### 📦 Stage 3-B — Distribution Analysis

> **What you see:** Four charts arranged in a 2×2 grid. Top row: Histogram of Total Withdrawals (left) and Histogram of Total Deposits (right), each with an overlaid box plot strip above the bars for simultaneous quartile/outlier reading. Bottom row: Box Plot for Withdrawals (left) and Box Plot for Deposits (right) in full vertical format showing minimum, Q1, median, Q3, maximum, and individual outlier dots.
>
> **Key observations:** The withdrawal histogram peaks sharply in the ₹40k–60k band with a long right tail extending past ₹80k — evidence of high-demand events (holidays, paydays). The deposit histogram is similarly bell-shaped but compressed entirely within ₹0–25k, confirming ATMs consistently dispense far more cash than they receive. The box plots make the outlier structure concrete — withdrawal outliers reach ₹100k+, deposit outliers ₹30k+.

![Distribution Analysis](Screenshot/Distribution%20analysis.png)

---

### 📅 Stage 3-C — Time-Based Trends (Full Daily Series)

> **What you see:** A dual-line time series spanning January 2022 – January 2024. The upper, more volatile line (darker blue) represents Total Withdrawals aggregated daily across all ATMs. The lower, calmer line (lighter blue) represents Total Deposits over the same period. The x-axis has quarterly date labels; the y-axis scales to ₹800k at peak.
>
> **Key observations:** Weekly oscillation is clearly visible in the withdrawal line — peaks every 5–7 days corresponding to weekends. Exceptional spikes reaching ₹600k–800k correspond to public holidays and salary dates. The deposit line remains comparatively flat and stable throughout — depositing behaviour is not seasonal or event-driven.

![Time Based Trends](Screenshot/Time%20Based%20Trends.png)

---

### 📅 Stage 3-D — Time-Based Trends (Day-of-Week & Time-of-Day Breakdown)

> **What you see:** Two colour-coded bar charts. Top chart: Average Withdrawals by Day of Week — a blue gradient where Monday is nearly white (lowest, ~₹49k average) and the bars darken progressively through the week, with Saturday and Sunday the darkest blue (highest, ~₹50k+). Bottom chart: Average Withdrawals by Time of Day — an orange gradient where Morning is the deepest orange (peak demand) and Afternoon is the lightest (lowest demand), with Evening and Night at intermediate levels.
>
> **Key observations:** The day-of-week gradient visually encodes the payday-and-weekend effect in a single glance. The time-of-day chart tells operations teams *when* to have ATMs fully stocked — Morning slots, before the start of business, must be prioritised.

![Time Based Trend](Screenshot/Time%20Based%20Trend.png)

---

### 🎉 Stage 3-E — Holiday & Event Impact and External Factors

> **What you see:** A four-chart panel. Top row: bar chart comparing average Total Withdrawals on Non-Holiday (blue bar, ~₹44k) vs Holiday (orange-red bar, ~₹46k) days (left); bar chart comparing No Event (blue, ~₹44k) vs Special Event (orange, ~₹45k) (right). Bottom row: a grouped box plot of Total Withdrawals by Weather Condition — Rainy / Clear / Cloudy / Snowy (left); a bar chart of average withdrawals by number of Nearby Competitor ATMs 0–5 (right).
>
> **Key observations:** Holidays produce a visible step-up in average withdrawals due to festive spending. Special events show a smaller but consistent lift. Weather boxplots have heavily overlapping IQRs — weather is not a planning variable. The competitor ATM chart is essentially flat — customers do not meaningfully shift ATM preference based on proximity to competition.

![Holiday Event Impact & External Factors](Screenshot/Holiday%20Event%20Impact%20%26%20External%20Factors.png)

---

### 🔗 Stage 3-F — Relationship Analysis

> **What you see:** Two charts side by side. Left: A scatter plot with Previous_Day_Cash_Level on the x-axis (₹20k–₹160k) and Cash_Demand_Next_Day on the y-axis (₹0–₹120k). A fitted linear trend line shows a slight negative slope. Right: A 5×5 correlation heatmap for all numeric features — Total_Withdrawals, Total_Deposits, Previous_Day_Cash_Level, Cash_Demand_Next_Day, Nearby_Competitor_ATMs. Cells are coloured from dark red (strong positive) to dark blue (strong negative); values are printed inside each cell.
>
> **Key observations:** The scatter plot confirms a weak negative relationship between leftover cash and next-day demand — ATMs that ran low yesterday need more tomorrow. The heatmap reveals the critical planning insight: the cell at Total_Withdrawals ↔ Cash_Demand_Next_Day is a deep red at **r = 0.895** — today's withdrawal volume is the single best predictor of tomorrow's replenishment need.

![Relationship Analysis](Screenshot/Relationship%20Analysis.png)

---

### 📐 Stage 4-A — Optimal Cluster Selection

> **What you see:** Two line charts side by side. Left: Elbow Method — Inertia vs k, with k on the x-axis (2–10) and inertia on the y-axis (₹8k–₹18k). A red dashed vertical line marks k=3. The blue curve drops steeply from k=2 to k=3 and then flattens to a gentle slope. Right: Silhouette Score vs k, with the orange line peaking at k=2 (~0.194), dipping at k=3 (~0.185), and fluctuating irregularly thereafter. A red dashed line again marks k=3.
>
> **Key observations:** The elbow is unambiguous at k=3 — additional clusters beyond 3 yield diminishing inertia reductions. The silhouette score technically peaks at k=2, but k=3 is selected because two clusters would merge the Supermarket and Gas Station ATM types into one undifferentiated group, losing operationally meaningful distinctions.

![Optimal Cluster Selection](Screenshot/Optimal%20Cluster%20Selection.png)

---

### 🗺️ Stage 4-B — Cluster Visualization (PCA Projection)

> **What you see:** A 2D scatter plot where each point is one ATM-transaction record projected onto two principal components (PC1 on x-axis, PC2 on y-axis). Points are coloured by cluster assignment — red (Cluster 0), blue (Cluster 1), green (Cluster 2). The legend identifies the three groups.
>
> **Key observations:** The three colour groups occupy distinct spatial regions in the PC space — red forms the upper region, green the lower-left, and blue the lower-right. This spatial separation confirms that K-Means found genuine structure in the data, not arbitrary partitions. The modest overlap at cluster boundaries reflects the real-world truth that some ATMs exhibit mixed behaviour patterns.

![Cluster Visualization (PCA Projection)](Screenshot/Cluster%20Visualization%20(PCA%20Projection).png)

---

### 📋 Stage 4-C — Cluster Profiles

> **What you see:** Three elements stacked vertically. Top: A summary table with columns Cluster / Avg Withdrawals / Avg Deposits / Avg Competitors / Top Location — showing three rows for clusters 0, 1, and 2. Middle: A bar chart titled "Average Withdrawals per Cluster" with three bars (red / blue / green) all reaching approximately ₹49.7k–49.9k, annotated with exact values. Bottom: A written "Cluster Interpretation Guide" bullet list explaining High / Medium / Low demand labels and corresponding refill strategies.
>
> **Key observations:** All three clusters have very similar average withdrawal volumes — the clustering is driven primarily by location context and competitive environment rather than raw cash volume. This means targeted strategies must focus on location type and time-of-year context rather than volume alone. The interpretation guide translates the technical cluster IDs into plain operational language that a branch manager can act on directly.

![Cluster Profiles](Screenshot/Cluster%20Profiles.png)

---

### 🔍 Stage 5-A — Anomaly Detection on Withdrawals

> **What you see:** At the top, a radio button to select detection method (IQR selected by default). Below that, three KPI cards: Lower Bound (IQR) = ₹8,474 / Upper Bound (IQR) = ₹91,127 / Anomalies Detected = 33. A multi-select widget allows choosing which ATMs to display (default: ATM_0001–ATM_0005). The main chart is a faceted scatter plot — one panel per selected ATM — showing Total Withdrawals over time. Normal transactions are blue dots; anomalous ones are red.
>
> **Key observations:** Anomalous red dots are sparse (33 out of 5,658 records = 0.6%) and distributed across the full timeline. Most appear at the extremes — either near ₹0 (possible machine fault or data error) or near/above ₹91k (exceptional demand). ATM_0003 shows a notable red spike around mid-2022. The faceted layout makes per-ATM patterns immediately comparable.

![Anomaly Detection on Withdrawal](Screenshot/Anomaly%20Detection%20on%20Withdrawal.png)

---

### 🗓️ Stage 5-B — Anomaly Map: Withdrawals by Date & Cluster

> **What you see:** A single large scatter chart. X-axis = Date (January 2022 – January 2024). Y-axis = Total Withdrawals (₹0 – ₹110k+). Colour encodes Anomaly Status (blue = Normal, orange = Anomaly for Cluster 0, red = Anomaly for Cluster 1, lighter red = Anomaly for Cluster 2). Shape encodes ATM Cluster (circle / square / diamond for Clusters 0 / 1 / 2). The vast majority of the chart is a dense blue cloud occupying the ₹20k–₹80k band.
>
> **Key observations:** Anomalies (coloured non-blue) appear at both the top of the chart (high-value outliers above ₹80k) and the bottom (near-zero outliers) — confirming two distinct anomaly causes: extreme demand events and potential machine/data failures. No seasonal clustering of anomalies is visible — they are distributed evenly across 2022 and 2023, ruling out a single systemic cause.

![Anomaly Map Withdrawals by Date & Cluster](Screenshot/Anomaly%20Map%20%20Withdrawals%20by%20Date%20%26%20Clus....png)

---

### 📍 Stage 5-C — Anomaly Rate by Location Type

> **What you see:** A horizontal bar chart with Location Type on the y-axis and Anomaly % on the x-axis (0–1.2%). Each bar is filled with a colour from a continuous red scale — dark crimson for the highest anomaly rate, light salmon for the lowest. Exact percentage labels appear on each bar.
>
> **Key observations:** Bank Branch leads at **1.1%** — more than double the rate of Mall and Gas Station (both 0.3%). Supermarket is second at 0.7% and Standalone at 0.5%. The colour gradient makes the risk ranking immediately readable at a glance. Operationally, this chart drives monitoring priority: Bank Branch ATMs require the most frequent audit cycles.

![Anomaly Rate by Location Type](Screenshot/Anomaly%20Rate%20by%20Location%20Type.png)

---

### 🎊 Stage 5-D — Anomaly Rates by Holiday & Event

> **What you see:** Two side-by-side bar charts. Left chart: "% Anomalous Days: Holidays vs Normal" — two bars, Non-Holiday (blue, 0.5%) and Holiday (orange-red, 2.4%). The holiday bar is nearly 5× taller, and its percentage label sits prominently above it. Right chart: "% Anomalous Days: Special Events vs Normal" — two bars, No Event (blue, 0.6%) and Special Event (orange, 0.5%). The two bars are almost equal height.
>
> **Key observations:** The holiday chart is the most actionable finding in the entire anomaly section — a 2.4% anomaly rate on holidays versus 0.5% on normal days represents a 4.8× multiplier. Cash management teams must treat every public holiday as a high-risk period requiring proactive pre-loading. Special events, by contrast, show a near-equal anomaly rate to normal days — suggesting they are better anticipated and already partially planned for.

![Anomaly Rates by Holiday & Event](Screenshot/Anomaly%20Rates%20by%20Holiday%20%26%20Event.png)

---

### ⚙️ Stage 6-A — Interactive Cash Demand Planner (Overview)

> **What you see:** The planner header with four KPI cards: Records (5,658), Unique ATMs (50), Anomalies (33), Anomaly Rate (0.6%). Below, two charts: on the left, a pie chart titled "Record Distribution by ATM Cluster" with three coloured slices — Cluster 0 red (41.7%), Cluster 2 blue (32.8%), Cluster 1 green (25.5%). On the right, a grouped box plot titled "Withdrawal Distribution per Cluster" showing three side-by-side box plots — all with similar medians (~₹47k–50k) but Cluster 0 showing the widest IQR and most outliers.
>
> **Key observations:** The pie chart confirms Cluster 0 (Supermarket ATMs) accounts for nearly half of all records — making it the operationally dominant group that deserves the most planning attention. The box plot comparison shows that while median withdrawals are similar, Cluster 0 has greater variability — meaning its demand is harder to predict and benefits most from safety stock buffers.

![Interactive Cash Demand Planner](Screenshot/Interactive%20Cash%20Demand%20Planner.png)

---

### 📄 Stage 6-B — Detailed Transaction Table

> **What you see:** A paginated data table with 12 visible columns: ATM_ID / Date / Day_of_Week / Time_of_Day / Location_Type / Total_Withdrawals / Total_Deposits / ATM_Cluster / Status / Holiday_Flag / Special_Event_Flag / Weather_Condition. The first visible rows are from 2022-01-01 (a Saturday). Examples: ATM_0042 at Mall with ₹50,864 withdrawals (Cluster 2, Normal, Holiday=1, Event=0, Snowy); ATM_0003 at Standalone with ₹28,640 withdrawals (Cluster 2, Normal, Holiday=1, Event=0, Cloudy).
>
> **Key observations:** The Status column makes anomaly identification instantaneous — any "Anomaly" label is immediately visible without needing to cross-reference a separate chart. Combining ATM_Cluster, Status, Holiday_Flag, and Special_Event_Flag in one row gives operations staff the full context for every transaction at a glance.

![Detailed Transaction Table](Screenshot/Detailed%20Transaction%20Table.png)

---

### 💡 Stage 6-C — Actionable Recommendations

> **What you see:** A green success alert banner at the top reading "Low anomaly rate (0.6%). Current cash management strategy appears sufficient." Below it, five bullet points with emoji icons covering: high-demand cluster refill priority (🏦), holiday pre-loading (🎉), special event coordination (🎸), clear weather weekend planning (☀️), and non-holiday anomaly investigation (🔍). A grey footer line states that the interactive planner integrates all FA-2 pipeline stages.
>
> **Key observations:** The green banner is dynamically generated — if the filtered anomaly rate exceeds a threshold, it would switch to a warning. The five recommendations are not generic — each maps directly to a quantitative finding from the EDA, clustering, or anomaly detection stages. This closes the loop from raw data to management action in a single workflow.

![Actionable Recommendations](Screenshot/Actionable%20Recommendations.png)

---

## 🔧 Installation & Local Run

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4 GB RAM minimum (8 GB recommended for comfortable performance)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/atm-demand-forecasting.git
cd atm-demand-forecasting

# 2. Create and activate a virtual environment (strongly recommended)
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Place the dataset in the project root
#    File must be named exactly: atm_cash_management_dataset.csv

# 5. Launch the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

### Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
scipy>=1.10.0
statsmodels>=0.14.0
```

---

## 🌐 Deployment

The app is hosted on **Streamlit Community Cloud** with zero-configuration deployment from this GitHub repository.

🔗 **Live URL:** [https://idai105-1000414-aditya-jitendra-kumar-sahani.streamlit.app/](https://idai105-1000414-aditya-jitendra-kumar-sahani.streamlit.app/)

**To deploy your own fork:**
1. Push this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → connect your repo → set `app.py` as the entry point
4. Upload `atm_cash_management_dataset.csv` to the repo or configure via Streamlit Secrets

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'statsmodels'` | `pip install statsmodels` |
| `FileNotFoundError` — dataset not found | Ensure `atm_cash_management_dataset.csv` is in the **project root** — same directory as `app.py` |
| Memory error with large dataset | Add `df = df.sample(n=100000, random_state=42)` at the top of `load_data()` |
| Port 8501 already in use | macOS/Linux: `lsof -i :8501` then `kill -9 <PID>` &nbsp;\|&nbsp; Windows: `netstat -ano \| findstr :8501` then `taskkill /PID <PID> /F` |
| Plotly charts not rendering | `pip install --upgrade plotly` |
| Streamlit version conflict | `pip install --upgrade streamlit` |
| `sklearn` import error | `pip install --upgrade scikit-learn` |

---

## ✅ Learning Outcomes

| Learning Outcome | Implementation |
|-----------------|---------------|
| **Exploratory Data Analysis** | 10+ Plotly visualisations — histograms, box plots, bar charts, scatter plots, correlation heatmaps, dual-axis time series |
| **Clustering Techniques** | K-Means with Elbow Method and Silhouette Score validation; PCA 2D projection for interpretability |
| **Anomaly Detection** | Dual-method approach — IQR statistical thresholds and Isolation Forest ML; contextualised by holiday, event, and location type |
| **Interactive Visualization** | Real-time Streamlit sidebar filters updating all charts, metrics, and the transaction table simultaneously |
| **Data-Driven Decision Making** | Actionable recommendations auto-generated from pipeline outputs with quantitative backing |
| **Python Development** | Modular `app.py` with `@st.cache_data`, Plotly Express, scikit-learn pipeline, well-commented code |
| **Version Control** | GitHub repository with proper structure, `.streamlit/config.toml`, and documentation |

---

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn — KMeans Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Scikit-learn — Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Scikit-learn — PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Plotly Express Tutorial](https://plotly.com/python/plotly-express/)
- [K-Means Clustering Guide — Neptune AI](https://neptune.ai/blog/k-means-clustering)
- [Anomaly Detection with IQR — KDNuggets](https://www.kdnuggets.com/)
- [Silhouette Score Explained — Towards Data Science](https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c)

---

## 👤 Submission Details

| Field | Value |
|-------|-------|
| **Student Name** | Aditya Jitendra Kumar Sahani|
| **Candidate Number** | 1000414 |
| **CRS Name** | Artificial Intelligence |
| **Course Name** | Data Mining |
| **Assessment** | Formative Assessment 2 (FA-2) |
| **Submission Date** | March 2026 |
| **GitHub Repository** | https://github.com/adityasahani392217/IDAI105-1000414-ADITYA-JITENDRA-KUMAR-SAHANI |
| **Live Application** | https://idai105-1000414-aditya-jitendra-kumar-sahani.streamlit.app/ |

---

<div align="center">

*🏧 ATM Demand Forecasting & Intelligence Dashboard*

*Mining Cash Patterns for Smarter Banking — FA-2 | Data Mining | 2026*

**Built with Python · Streamlit · scikit-learn · Plotly**

</div>