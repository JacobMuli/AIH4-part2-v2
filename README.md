🌱 NDVI-Based Potato Yield Intelligence (Option A v4)

This project provides an AI-powered forecasting system for potato yield using NDVI vegetation health indicators, historical production, and engineered agronomic features. It also includes SHAP and LIME explainability, cold-storage planning, and multi-county comparison dashboards.

🚀 Key Features
1. Yield Forecasting (Recursive Multi-Year Model)

Predict yield (t/ha) using NDVI + lag features + agronomic time features

Predict production (tonnes) using forecasted yield × area

Supports area override (custom user input)

Forecast multiple years ahead using recursive model updates

2. Explainability Dashboards
🔍 SHAP Explainability

Global summary bar plot

SHAP dot plot

Automated human-readable SHAP insights

Multi-County SHAP comparison dashboard

🟩 LIME Explainability

Local per-observation model explanations

Visual breakdown showing positive/negative influence

3. Cold Storage Recommendation Engine

Based on predicted production:

Allocates storage across Type 1 / Type 2 / Type 3 chambers

Maximizes capacity utilization

Ensures scalable storage planning for FPOs & county planners

4. Upload-Ready Architecture

You can run the app with:

Uploaded CSV + model

Or preloaded artifacts in /mnt/data

🧠 Model Training (Random Forest v2)

The model uses:

Mean annual NDVI

Yield & NDVI lag features

Rolling averages

Year index (temporal structure)

County one-hot encoding

The training script:

Cleans agronomic inconsistencies

Builds engineered features

Performs year-based train/test split

Produces RandomForest model

Exports:

ndvi_optionb_cleaned.csv

rf_optionb_ndvi_model_v2.joblib

training_results.json

🖥 How to Run Locally
pip install -r requirements.txt
streamlit run app_v4.py

📦 Repo Structure
/
├── app_v4.py
├── ndvi_optionb_cleaned.csv
├── rf_optionb_ndvi_model_v2.joblib
├── training_script.py
├── requirements.txt
└── README.md

🎯 Intended Users

County Agriculture Directorates

FPOs / Cooperatives

Cold-storage investors

Policy planners

Machine Learning practitioners in agriculture
