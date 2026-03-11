# 🏧 ATM Demand Forecasting & Intelligence Dashboard

## 📋 Overview

An interactive data mining application for FinTrust Bank Ltd. that transforms raw ATM transaction data into actionable business insights. This dashboard performs comprehensive exploratory data analysis (EDA), clusters ATMs based on demand behavior, detects anomalies during holidays/events, and provides an interactive planning tool for cash management optimization.

**Live Demo:** [https://45ozrhhclerxxenvhikere.streamlit.app/](https://45ozrhhclerxxenvhikere.streamlit.app/)



---

## ✨ Key Features

### 📊 Exploratory Data Analysis
| Feature | Description |
|---------|-------------|
| **Distribution Analysis** | Histograms and box plots for withdrawals/deposits with statistical summaries |
| **Temporal Patterns** | Daily trends, day-of-week patterns, and time-of-day analysis with peak hour identification |
| **Holiday Impact** | Comparative analysis of withdrawal patterns on holidays vs normal days |
| **Event Analysis** | Special event impact assessment (concerts, sports, festivals) |
| **External Factors** | Weather condition impact and competitor influence analysis |
| **Relationship Analysis** | Correlation heatmaps and scatter plots with trend lines |

### 📈 ATM Clustering
- **K-Means Clustering**: Groups ATMs into meaningful segments based on demand behavior
- **Optimal Cluster Selection**: Elbow method and silhouette score analysis for k selection
- **PCA Visualization**: 2D projection of ATM clusters for intuitive understanding
- **Cluster Profiles**: Interpretable summaries with key characteristics:
  - 🏙️ **High-demand**: Urban locations, high traffic, fewer competitors
  - 🏢 **Steady-demand**: Business hubs, consistent patterns
  - 🏡 **Low-demand**: Rural areas, seasonal usage

### 🚨 Anomaly Detection
- **Statistical Methods**: IQR-based outlier detection with dynamic thresholds
- **Machine Learning**: Isolation Forest for contextual anomaly detection
- **Holiday/Event Analysis**: Compares anomaly rates during special days
- **Visual Highlighting**: Time series with anomalies marked in red
- **Pattern Recognition**: Identifies unusual spikes that may indicate fraud or system issues

### ⚙️ Interactive Planner
- **Dynamic Filtering**: Filter by day, time, location, holidays, and events
- **Real-time Updates**: All visualizations and metrics update instantly
- **Cluster Assignment**: View ATM classifications with each transaction
- **Anomaly Flags**: Identify unusual transactions at a glance
- **Actionable Insights**: Data-driven recommendations for:
  - Cash loading schedules
  - Refill frequency optimization
  - Inventory management
  - Risk assessment

---

## 🖥️ App Deployment

### Live Web App
The application is hosted on Streamlit Community Cloud:
👉 **[Launch Live App](https://45ozrhhclerxxenvhikere.streamlit.app/)**

### Local Installation

#### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional but recommended)
- 4GB RAM minimum (8GB recommended for large datasets)

#### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/atm-demand-forecasting.git
   cd atm-demand-forecasting
Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
2.**Install Dependencies**

bash
`pip install -r requirements.txt`
Add Dataset

```bash
# Place your ATM dataset CSV file in the project directory
# Default filename: atm_cash_management_dataset.csv
# Ensure it contains required columns
```
3.**Run the App**

bash
`streamlit run app.py`
The app will open in your default browser at http://localhost:8501

## 📁 Project Structure
text
```
atm-demand-forecasting/
│
├── app.py                          # Main Streamlit application (2,000+ lines)
├── requirements.txt                 # Python dependencies with versions
├── atm_cash_management_dataset.csv  # ATM transaction data (sample)
├── README.md                        # Project documentation
│
├── .streamlit/                       # Streamlit configuration
│   └── config.toml                   # Theme and server settings
│       ├── primaryColor = "#FF4B4B"
│       └── backgroundColor = "#FFFFFF"
│
├── assets/                           # Images and resources
│   ├── screenshot.png                 # App screenshot
│   └── demo.gif                       # App demo animation
│
└── notebooks/                         # Jupyter notebooks (optional)
    └── eda_analysis.ipynb             # Initial exploration
```
## 🎯 Learning Outcomes Achieved
This project demonstrates proficiency in:

Learning Outcome	Implementation
✅ Exploratory Data Analysis	10+ visualizations with statistical insights
✅ Clustering Techniques	K-Means with elbow method and silhouette analysis
✅ Anomaly Detection	IQR and Isolation Forest with holiday/event context
✅ Interactive Visualization	Real-time filtering with Streamlit and Plotly
✅ Data-Driven Decision Making	Actionable recommendations based on patterns
✅ Python Development	Modular, well-commented, reproducible code
✅ Version Control	GitHub repository with proper documentation
## 📊 Dataset Requirements
The app expects a CSV file with the following columns:

Column Name	Data Type	Description	Example
ATM_ID	String	Unique identifier for each ATM	ATM_0041
Date	Date	Transaction date	2022-04-25
Day_of_Week	String	Monday through Sunday	Monday
Time_of_Day	String	Morning, Afternoon, Evening, Night	Morning
Total_Withdrawals	Integer	Cash withdrawn	57450
Total_Deposits	Integer	Cash deposited	9308
Location_Type	String	Urban, Semi-Urban, Rural, etc.	Standalone
Holiday_Flag	Binary	1 for holiday, 0 otherwise	0
Special_Event_Flag	Binary	1 for special event, 0 otherwise	0
Previous_Day_Cash_Level	Integer	Starting cash inventory	112953
Weather_Condition	String	Clear, Rainy, Snowy, Cloudy	Rainy
Nearby_Competitor_ATMs	Integer	Number of competing ATMs nearby	5
Cash_Demand_Next_Day	Integer	Target variable for forecasting	44165
## 🔧 Troubleshooting Guide
### Common Issues and Solutions
#### 1. ModuleNotFoundError: No module named 'statsmodels'
bash
`pip install statsmodels`
#### 2. FileNotFoundError: Dataset not found
```bash
# Check current directory
ls -la  # Linux/Mac
dir     # Windows

# Ensure file is in correct location
mv your_dataset.csv atm_cash_management_dataset.csv
```
#### 3. Memory Issues with Large Datasets
```python
# Add to app.py for data sampling
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)
    st.warning("⚠️ Dataset sampled to 100,000 rows for performance")
```
#### 4. Streamlit Port Already in Use
```bash
# Kill process using port 8501
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8501
kill -9 <PID>
```
#### 5. Plotly Rendering Issues
```bash
# Update plotly
pip install --upgrade plotly
```
## 🚀 Performance Optimization
Caching: @st.cache_data decorators for data loading and expensive computations

Lazy Loading: Visualizations render only when tabs are selected

Data Sampling: Optional sampling for large datasets

Efficient Filtering: Pandas boolean indexing for real-time updates

## 📝 License
This project is licensed under the MIT License 


Copyright (c) 2026 [Mann Paresh Patel]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
## 👨‍💻 Author
Mann Paresh Patel
WACP Candidate Number- 1000428
Data Mining - Formative Assessment 2

## 🙏 Acknowledgments
FinTrust Bank Ltd. - For providing the project scenario and dataset

Streamlit Team - For the amazing web app framework

Scikit-learn - For machine learning algorithms

Plotly - For interactive visualizations

Course Instructors - For guidance and feedback

## 📚 References
Streamlit Documentation

Scikit-learn Clustering

Plotly Express Tutorial

Anomaly Detection Guide

## 📧 Contact & Support
For questions, feedback, or issues:

GitHub Issues: Create an issue

## 🗓️ Changelog
Version 1.0.0 (March 2026)
Initial release with complete FA-2 functionality

EDA with 10+ visualizations

K-Means clustering with elbow method

Anomaly detection with IQR and Isolation Forest

Interactive planner with real-time filtering

###Made for Data Mining FA-2
© 2026 All Rights Reserved   
