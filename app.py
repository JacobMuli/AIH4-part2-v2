import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
import shap
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path

st.set_page_config(page_title="AI Yield Intelligence", layout="wide")
st.title("🌱 Cold_Sto Forecasting Engine")

DEFAULT_CSV = "/mnt/data/ndvi_optionb_cleaned.csv"
DEFAULT_MODEL = "/mnt/data/rf_optionb_ndvi_model_v2.joblib"

# -----------------------------------------------------------------------------
# Load helpers
# -----------------------------------------------------------------------------
@st.cache_data
def load_df(src):
    df = pd.read_csv(src)
    df.columns = df.columns.str.lower().str.strip()
    return df

@st.cache_resource
def load_rf_model(src):
    return joblib.load(src)

# -----------------------------------------------------------------------------
# Upload or use defaults
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Upload Data & Model")
upload_csv = st.sidebar.file_uploader("Upload Cleaned NDVI CSV", type=["csv"])
upload_model = st.sidebar.file_uploader("Upload Model (.joblib)", type=["joblib"])

csv_path = upload_csv if upload_csv else (DEFAULT_CSV if Path(DEFAULT_CSV).exists() else None)
model_path = upload_model if upload_model else (DEFAULT_MODEL if Path(DEFAULT_MODEL).exists() else None)

if csv_path is None or model_path is None:
    st.warning("Please upload both NDVI CSV & model or place them under /mnt/data/")
    st.stop()

df = load_df(csv_path)
rf = load_rf_model(model_path)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
page = st.sidebar.radio(
    "📌 Navigation",
    ["Forecasting", "SHAP Explainability", "LIME Explainability",
     "Multi-County SHAP Comparison", "Storage Planner", "Downloads"]
)

# -----------------------------------------------------------------------------
# County selection & area override
# -----------------------------------------------------------------------------
counties = sorted(df["admin_1"].dropna().unique())
county = st.sidebar.selectbox("Select County", counties)

default_area = float(df[df["admin_1"] == county]["area"].mean())
area_override = st.sidebar.number_input(
    "Area to Forecast (ha)",
    min_value=1.0,
    value=default_area,
    step=1.0
)

cdf = df[df["admin_1"] == county].sort_values("harvest_year").reset_index(drop=True)

# -----------------------------------------------------------------------------
# Feature preparation (SHAP, LIME, Forecasting)
# -----------------------------------------------------------------------------
def prepare_features(df_rows, area_value=None):
    X = df_rows.copy().reset_index(drop=True)
    X_feat = pd.DataFrame()

    # Base numeric features
    X_feat["mean_annual_ndvi"] = X["mean_annual_ndvi"].astype(float)
    X_feat["area"] = area_value if area_value else X["area"]

    # planting month FIXED (prevents crash)
    if "planting_month" in X.columns:
        X_feat["planting_month"] = X["planting_month"].fillna(1).astype(int)
    else:
        X_feat["planting_month"] = 1

    # year index
    if "harvest_year" in X.columns:
        X_feat["year_index"] = X["harvest_year"] - X["harvest_year"].min()
    else:
        X_feat["year_index"] = 0

    lag_features = [
        "yield_lag_1","yield_lag_2","yield_lag_3","yield_roll3",
        "ndvi_lag_1","ndvi_lag_2","ndvi_lag_3","ndvi_roll3","ndvi_change"
    ]
    for lf in lag_features:
        X_feat[lf] = X[lf] if lf in X.columns else 0.0

    # County dummy features (if model was trained with them)
    if hasattr(rf, "feature_names_in_"):
        feats = list(rf.feature_names_in_)
        for f in feats:
            if f.startswith("adm1_"):
                cname = f.replace("adm1_", "")
                X_feat[f] = (X["admin_1"] == cname).astype(int)
        X_feat = X_feat.reindex(columns=feats, fill_value=0.0)

    return X_feat

# -----------------------------------------------------------------------------
# Recursive Multi-Year Forecast (FIXED)
# -----------------------------------------------------------------------------
def recursive_forecast(agg_df, model, steps, area_value):
    hist = agg_df.copy().reset_index(drop=True)
    predictions = []

    for _ in range(steps):
        last = hist.iloc[-1].copy()

        # Create missing fields required by model
        last["planting_month"] = 1   # FIXED (avoid crash)

        # lags
        for lag in [1, 2, 3]:
            last[f"yield_lag_{lag}"] = hist["yield"].iloc[-lag] if len(hist) >= lag else hist["yield"].iloc[-1]
            last[f"ndvi_lag_{lag}"] = hist["mean_annual_ndvi"].iloc[-lag] if len(hist) >= lag else hist["mean_annual_ndvi"].iloc[-1]

        last["yield_roll3"] = hist["yield"].tail(3).mean()
        last["ndvi_roll3"] = hist["mean_annual_ndvi"].tail(3).mean()
        last["ndvi_change"] = last["mean_annual_ndvi"] - last["ndvi_lag_1"]

        next_year = int(last["harvest_year"]) + 1
        last["year_index"] = next_year - agg_df["harvest_year"].min()

        # Build feature vector
        X_next = prepare_features(pd.DataFrame([last]), area_value)
        pred_y = float(model.predict(X_next)[0])
        pred_prod = pred_y * area_value

        predictions.append({
            "harvest_year": next_year,
            "predicted_yield": pred_y,
            "predicted_production": pred_prod
        })

        new_row = {
            "harvest_year": next_year,
            "area": area_value,
            "production": pred_prod,
            "mean_annual_ndvi": last["mean_annual_ndvi"],
            "yield": pred_y,
            "admin_1": last["admin_1"],
            "planting_month": 1  # FIXED
        }

        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(predictions)

# -----------------------------------------------------------------------------
# PAGE: Forecasting
# -----------------------------------------------------------------------------
if page == "Forecasting":
    st.header(f"📈 Forecasting — {county}")
    st.dataframe(cdf.head())

    agg = cdf.groupby("harvest_year").agg({
        "area": "mean",
        "production": "sum",
        "mean_annual_ndvi": "mean",
        "yield": "mean"
    }).reset_index()
    agg["admin_1"] = county
    agg["planting_month"] = 1  # required for features

    last_year = int(agg["harvest_year"].max())
    current_year = datetime.datetime.now().year

    target_year = st.number_input(
        "Forecast up to year:",
        min_value=last_year + 1, max_value=current_year, value=last_year + 1
    )

    steps = target_year - last_year

    forecast_df = recursive_forecast(agg, rf, steps, area_override)

    st.subheader("Forecast Results")
    st.dataframe(forecast_df)

    final_prod = forecast_df.iloc[-1]["predicted_production"]
    st.metric("Final Year Predicted Production (tonnes)", f"{final_prod:,.1f}")

# -----------------------------------------------------------------------------
# PAGE: SHAP Explainability
# -----------------------------------------------------------------------------
elif page == "SHAP Explainability":
    st.header("🔍 SHAP Explainability")

    X_shap = prepare_features(cdf, area_override)
    explainer = shap.TreeExplainer(rf)

    # sample
    X_sample = X_shap.sample(min(len(X_shap), 300), random_state=42)

    # Summary bar
    fig = plt.figure(figsize=(9, 5))
    shap.summary_plot(explainer.shap_values(X_sample), X_sample, plot_type="bar", show=False)
    st.pyplot(fig)

    # Dot plot
    fig = plt.figure(figsize=(9, 5))
    shap.summary_plot(explainer.shap_values(X_sample), X_sample, show=False)
    st.pyplot(fig)

    # Auto Summary
    abs_vals = np.abs(explainer.shap_values(X_sample)).mean(axis=0)
    ordered = sorted(zip(X_sample.columns, abs_vals), key=lambda x: x[1], reverse=True)

    st.subheader("📘 Automated SHAP Summary")
    for i, (feat, score) in enumerate(ordered[:10], 1):
        st.markdown(f"- **{i}. {feat}** — mean(|SHAP|) = **{score:.4f}**")

# -----------------------------------------------------------------------------
# PAGE: LIME Explainability
# -----------------------------------------------------------------------------
elif page == "LIME Explainability":
    st.header("🟩 LIME Explainability")

    X_lime = prepare_features(cdf, area_override)
    X_vals = X_lime.values

    explainer = LimeTabularExplainer(
        X_vals,
        feature_names=X_lime.columns.tolist(),
        mode="regression"
    )

    idx = st.number_input(
        "Select row index:", min_value=0,
        max_value=len(X_vals)-1, value=0
    )

    exp = explainer.explain_instance(X_vals[int(idx)], rf.predict, num_features=10)

    st.subheader("Top LIME Feature Contributions")
    st.json(exp.as_list())

    st.pyplot(exp.as_pyplot_figure())

# -----------------------------------------------------------------------------
# PAGE: Multi-County SHAP Comparison
# -----------------------------------------------------------------------------
elif page == "Multi-County SHAP Comparison":
    st.header("🌍 Multi-County SHAP Comparison")

    c1 = st.selectbox("County A", counties, index=0)
    c2 = st.selectbox("County B", counties, index=1)

    dfA = df[df["admin_1"] == c1]
    dfB = df[df["admin_1"] == c2]

    XA = prepare_features(dfA, area_override)
    XB = prepare_features(dfB, area_override)

    explainer = shap.TreeExplainer(rf)
    svA = np.abs(explainer.shap_values(XA)).mean(axis=0)
    svB = np.abs(explainer.shap_values(XB)).mean(axis=0)

    comp = pd.DataFrame({
        "feature": XA.columns,
        f"{c1}": svA,
        f"{c2}": svB
    }).sort_values(c1, ascending=False)

    st.dataframe(comp)

# -----------------------------------------------------------------------------
# PAGE: Storage Planner
# -----------------------------------------------------------------------------
elif page == "Storage Planner":
    st.header("❄ Cold Storage Planner")

    tonnes = st.number_input("Enter production (tonnes):", min_value=0.0, value=0.0)

    def storage_plan(tonnes):
        sizes = [1000, 500, 250]
        allocation = []
        rem = tonnes
        type_id = 1

        for s in sizes:
            count = int(rem // s)
            if count > 0:
                allocation.append({f"type {type_id}": {"size": s, "count": count}})
                rem -= count * s
            type_id += 1

        if rem > 0:
            allocation.append({f"type {type_id}": {"size": 250, "count": 1}})

        return allocation

    if tonnes > 0:
        st.json(storage_plan(tonnes))

# -----------------------------------------------------------------------------
# PAGE: Download Models/Data
# -----------------------------------------------------------------------------
elif page == "Downloads":
    st.header("⬇ Downloads")

    if Path(DEFAULT_CSV).exists():
        st.download_button("Download Cleaned CSV",
                           open(DEFAULT_CSV, "rb").read(),
                           file_name="ndvi_optionb_cleaned.csv")

    if Path(DEFAULT_MODEL).exists():
        st.download_button("Download Model",
                           open(DEFAULT_MODEL, "rb").read(),
                           file_name="rf_optionb_ndvi_model_v2.joblib")

st.caption("🔥 Option A v4 — Enhanced Forecasting + SHAP + LIME + Storage Engine")

# # app_v3.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import datetime
# import shap
# import os
# from pathlib import Path

# # Optional LIME import removed to keep this SHAP-focused v3.
# # If later needed, import lime.lime_tabular

# st.set_page_config(page_title="Option A v3 — NDVI Yield Forecasting + SHAP", layout="wide")
# st.title("🌱 Option A v3 — NDVI Potato Yield Forecasting + SHAP Explainability")

# # -------------------------
# # Default local paths (useful if you placed artifacts into /mnt/data)
# # -------------------------
# DEFAULT_CSV = "/mnt/data/ndvi_optionb_cleaned.csv"
# DEFAULT_MODEL = "/mnt/data/rf_optionb_ndvi_model_v2.joblib"

# # -------------------------
# # Sidebar: Uploads and navigation
# # -------------------------
# st.sidebar.header("Data & Model")
# data_file = st.sidebar.file_uploader("Upload cleaned NDVI CSV (ndvi_optionb_cleaned.csv)", type=["csv"])
# model_file = st.sidebar.file_uploader("Upload trained model (joblib)", type=["joblib"])

# # If user didn't upload, but file exists in default path, use it
# use_default_data = False
# use_default_model = False
# if data_file is None and Path(DEFAULT_CSV).exists():
#     use_default_data = True
# if model_file is None and Path(DEFAULT_MODEL).exists():
#     use_default_model = True

# if (not data_file and not use_default_data) or (not model_file and not use_default_model):
#     st.info("Please upload dataset and model, or place them at the default locations (/mnt/data).")
#     st.stop()

# # Load dataset (prefer upload)
# @st.cache_data(show_spinner=False)
# def load_dataframe(uploaded, default_path):
#     if uploaded:
#         df = pd.read_csv(uploaded)
#     else:
#         df = pd.read_csv(default_path)
#     df.columns = df.columns.str.lower().str.strip()
#     return df

# @st.cache_resource(show_spinner=False)
# def load_model(uploaded, default_path):
#     if uploaded:
#         return joblib.load(uploaded)
#     else:
#         return joblib.load(default_path)

# df = load_dataframe(data_file if data_file is not None else None, DEFAULT_CSV if use_default_data else None)
# rf = load_model(model_file if model_file is not None else None, DEFAULT_MODEL if use_default_model else None)

# # -------------------------
# # County & area selection
# # -------------------------
# if "admin_1" not in df.columns:
#     st.error("Dataset missing 'admin_1' column. Please provide a cleaned NDVI CSV with county column 'admin_1'.")
#     st.stop()

# counties = sorted(df["admin_1"].dropna().unique())
# county = st.sidebar.selectbox("Select County", counties)

# # default area: mean of selected county (fallback to global mean)
# default_area = float(df[df["admin_1"] == county]["area"].mean()) if "area" in df.columns else 100.0
# area_override = st.sidebar.number_input(
#     "Enter Area to Forecast (ha)",
#     min_value=1.0,
#     value=default_area,
#     step=1.0,
#     help="Overrides dataset area for forecasting (used for production calculation)."
# )

# # Navigation pages
# page = st.sidebar.radio("Navigation", ["Forecasting", "SHAP Explainability", "Multi-County SHAP Comparison", "Storage Planner", "Download"])

# # -------------------------
# # Global helper: prepare features (used in SHAP & forecasting)
# # -------------------------
# def prepare_shap_features(df_rows, area_override_val):
#     """
#     Build the feature matrix expected by the model from raw df rows.
#     - ensures NDVI, area, planting_month, year_index, and county dummies are present
#     - orders columns to match model.feature_names_in_ if available
#     """
#     X = df_rows.copy().reset_index(drop=True)

#     # base numeric features
#     X_feat = pd.DataFrame()
#     X_feat["mean_annual_ndvi"] = X["mean_annual_ndvi"].astype(float)
#     X_feat["area"] = area_override_val if area_override_val is not None else X.get("area", 0.0)
#     X_feat["planting_month"] = X.get("planting_month", 1).fillna(1).astype(int)
#     # year_index: if harvest_year present, use offset, else 0
#     if "harvest_year" in X.columns:
#         X_feat["year_index"] = X["harvest_year"] - X["harvest_year"].min()
#     else:
#         X_feat["year_index"] = 0

#     # include lag / rolling features if present in dataset (we trained with them)
#     optional_cols = [
#         "yield_lag_1","yield_lag_2","yield_lag_3","yield_roll3",
#         "ndvi_lag_1","ndvi_lag_2","ndvi_lag_3","ndvi_roll3","ndvi_change"
#     ]
#     for c in optional_cols:
#         if c in X.columns:
#             X_feat[c] = X[c].fillna(0.0).astype(float)
#         else:
#             X_feat[c] = 0.0

#     # county dummies if model expects them - create dummies for admin_1 values present in X
#     if hasattr(rf, "feature_names_in_"):
#         feature_list = list(rf.feature_names_in_)
#         # create dummy columns for any "adm1_" features
#         adm_feats = [f for f in feature_list if f.startswith("adm1_")]
#         for adm in adm_feats:
#             name = adm.replace("adm1_", "")
#             # if admin_1 column exists, set 1 where matched
#             if "admin_1" in X.columns:
#                 X_feat[adm] = (X["admin_1"] == name).astype(int)
#             else:
#                 X_feat[adm] = 0
#         # reorder to match model feature order
#         X_feat = X_feat.reindex(columns=feature_list, fill_value=0.0)

#     return X_feat

# # -------------------------
# # Recursive forecasting (updates lags & rolls each step)
# # -------------------------
# def recursive_forecast_by_county(agg_df, rf_model, features_list, years_ahead=1, area_override_val=None):
#     """
#     agg_df: per-year aggregated DataFrame for a single county (harvest_year, area, production, mean_annual_ndvi, yield, admin_1)
#     features_list: list of features in the order model expects (if None, infer from rf.feature_names_in_)
#     """
#     hist = agg_df.sort_values("harvest_year").reset_index(drop=True).copy()
#     preds = []

#     # ensure features list
#     if features_list is None and hasattr(rf_model, "feature_names_in_"):
#         features_list = list(rf_model.feature_names_in_)
#     elif features_list is None:
#         # fallback to default set used in training
#         features_list = [
#             "mean_annual_ndvi","area","planting_month","year_index",
#             "yield_lag_1","yield_lag_2","yield_lag_3","yield_roll3",
#             "ndvi_lag_1","ndvi_lag_2","ndvi_lag_3","ndvi_roll3","ndvi_change"
#         ]

#     for step in range(years_ahead):
#         last = hist.iloc[-1].copy()

#         # compute lags from hist
#         for lag in [1,2,3]:
#             last[f"yield_lag_{lag}"] = hist["yield"].iloc[-lag] if len(hist) >= lag else hist["yield"].iloc[-1]
#             last[f"ndvi_lag_{lag}"] = hist["mean_annual_ndvi"].iloc[-lag] if len(hist) >= lag else hist["mean_annual_ndvi"].iloc[-1]
#         last["yield_roll3"] = hist["yield"].iloc[-3:].mean() if len(hist) >= 3 else hist["yield"].mean()
#         last["ndvi_roll3"] = hist["mean_annual_ndvi"].iloc[-3:].mean() if len(hist) >= 3 else hist["mean_annual_ndvi"].mean()
#         last["ndvi_change"] = last["mean_annual_ndvi"] - last.get("ndvi_lag_1", last["mean_annual_ndvi"])
#         last["year_index"] = last["harvest_year"] - agg_df["harvest_year"].min() + (step + 1)

#         # build feature vector in required order
#         xrow = []
#         for feat in features_list:
#             if feat in last:
#                 xrow.append(last[feat])
#             else:
#                 # county dummies
#                 if feat.startswith("adm1_"):
#                     xrow.append(1.0 if feat == f"adm1_{last['admin_1']}" else 0.0)
#                 else:
#                     xrow.append(0.0)
#         xarr = np.array([xrow], dtype=float)

#         pred_y = float(rf_model.predict(xarr)[0])
#         area_use = area_override_val if area_override_val is not None else last["area"]
#         pred_prod = pred_y * area_use
#         next_year = int(last["harvest_year"]) + 1

#         preds.append({"harvest_year": next_year, "predicted_yield": float(pred_y), "predicted_production": float(pred_prod)})

#         # append predicted row to hist for next iteration
#         new_hist = {
#             "harvest_year": next_year,
#             "area": area_use,
#             "production": pred_prod,
#             "mean_annual_ndvi": last["mean_annual_ndvi"],  # NDVI kept stable (you can build NDVI forecast later)
#             "yield": pred_y,
#             "admin_1": last["admin_1"]
#         }
#         hist = pd.concat([hist, pd.DataFrame([new_hist])], ignore_index=True)

#     return pd.DataFrame(preds)

# # -------------------------
# # Storage packer with Type labels
# # -------------------------
# def pack_storage(tonnes, sizes=[1000, 500, 250], fill=0.9):
#     needed = int(np.ceil(tonnes / fill))
#     allocation = []
#     remaining = needed

#     for s in sizes:
#         c = remaining // s
#         if c > 0:
#             allocation.append({"size": s, "count": int(c)})
#             remaining -= c * s

#     if remaining > 0:
#         allocation.append({"size": 250, "count": 1})

#     typed_allocation = {f"type {i+1}": allocation[i] for i in range(len(allocation))}
#     total_capacity = sum(a["size"] * a["count"] for a in allocation)
#     utilization = float(tonnes / total_capacity) if total_capacity > 0 else 0.0

#     return {"required_capacity": needed, "allocation": typed_allocation, "utilization": utilization}

# # -------------------------
# # PAGE: Forecasting
# # -------------------------
# if page == "Forecasting":
#     st.header(f"Forecasting — County: {county}")

#     # preview
#     st.subheader("Dataset preview (filtered)")
#     cdf = df[df["admin_1"] == county].sort_values("harvest_year").reset_index(drop=True)
#     st.dataframe(cdf.head(10))

#     # Aggregation per year (if multiple rows per year)
#     agg = cdf.groupby("harvest_year").agg({
#         "area": "sum",
#         "production": "sum",
#         "mean_annual_ndvi": "mean",
#         "yield": "mean"
#     }).reset_index()
#     agg["admin_1"] = county

#     if agg.empty:
#         st.warning("No aggregated rows for this county — cannot forecast.")
#     else:
#         st.subheader("Multi-year recursive forecasting")
#         last_year = int(agg["harvest_year"].max())
#         current_year = datetime.datetime.now().year

#         target = st.number_input(
#             "Forecast up to year (inclusive)",
#             min_value=last_year + 1,
#             max_value=current_year,
#             value=last_year + 1,
#             step=1
#         )

#         years_ahead = int(target - last_year)
#         FEATURES = list(rf.feature_names_in_) if hasattr(rf, "feature_names_in_") else None

#         with st.spinner("Running recursive forecast..."):
#             forecast_df = recursive_forecast_by_county(agg, rf, FEATURES, years_ahead=years_ahead, area_override_val=area_override)

#         st.subheader("Forecast results (per year)")
#         st.dataframe(forecast_df)

#         final_tonnes = float(forecast_df.iloc[-1]["predicted_production"])
#         st.metric("Final Year Predicted Yield (t/ha)", f"{forecast_df.iloc[-1]['predicted_yield']:.2f}")
#         st.metric("Final Year Predicted Production (tonnes)", f"{final_tonnes:,.0f}")

#         # storage
#         st.subheader("Storage recommendation (based on final year production)")
#         storage = pack_storage(final_tonnes)
#         st.json(storage)

# # -------------------------
# # PAGE: SHAP Explainability
# # -------------------------
# elif page == "SHAP Explainability":
#     st.header("🔍 SHAP Explainability (Single-county)")

#     # filtered county df
#     cdf = df[df["admin_1"] == county].sort_values("harvest_year").reset_index(drop=True)

#     if cdf.shape[0] < 2:
#         st.warning("Not enough rows for SHAP analysis for this county.")
#     else:
#         X_shap = prepare_shap_features(cdf, area_override)
#         # compute SHAP values (sampled)
#         explainer = shap.TreeExplainer(rf)
#         sample_n = min(500, X_shap.shape[0])
#         X_sample = X_shap.sample(sample_n, random_state=42)

#         st.subheader("Global feature importance (SHAP summary bar)")
#         fig = plt.figure(figsize=(8,5))
#         shap.summary_plot(explainer.shap_values(X_sample), X_sample, plot_type="bar", show=False)
#         st.pyplot(fig)

#         st.subheader("Feature impact distribution (SHAP dot plot)")
#         fig = plt.figure(figsize=(8,5))
#         shap.summary_plot(explainer.shap_values(X_sample), X_sample, show=False)
#         st.pyplot(fig)

#         # Auto summary text
#         mean_abs = np.abs(explainer.shap_values(X_sample)).mean(axis=0)
#         feat_scores = dict(zip(X_sample.columns, mean_abs))
#         ordered = sorted(feat_scores.items(), key=lambda x: x[1], reverse=True)

#         st.subheader("📝 SHAP Auto-summary")
#         total = sum(mean_abs) if mean_abs.sum() != 0 else 1.0
#         for i, (f, s) in enumerate(ordered[:8], start=1):
#             pct = s / total * 100
#             st.markdown(f"- **{i}. {f}** — contributes approximately **{pct:.1f}%** (mean|SHAP|={s:.4f})")

#         st.markdown(
#             """
# **Interpretation guides**
# - Positive SHAP values for a feature push the model to predict **higher yields**.
# - Negative SHAP values push predicted yield **lower**.
# - Lag features indicate model reliance on historical yields.
# - Large NDVI importance suggests vegetation health strongly influences yield predictions.
# """
#         )

# # -------------------------
# # PAGE: Multi-County SHAP Comparison
# # -------------------------
# elif page == "Multi-County SHAP Comparison":
#     st.header("🏳️‍🌍 Multi-County SHAP Comparison")

#     county_list = sorted(df["admin_1"].dropna().unique())
#     c1 = st.selectbox("County A", county_list, index=0)
#     c2 = st.selectbox("County B", county_list, index=min(1, len(county_list)-1))

#     dfA = df[df["admin_1"] == c1].sort_values("harvest_year").reset_index(drop=True)
#     dfB = df[df["admin_1"] == c2].sort_values("harvest_year").reset_index(drop=True)

#     if dfA.shape[0] < 2 or dfB.shape[0] < 2:
#         st.warning("Not enough rows in one of the selected counties for robust SHAP summary.")
#     else:
#         XA = prepare_shap_features(dfA, area_override)
#         XB = prepare_shap_features(dfB, area_override)
#         explainer = shap.TreeExplainer(rf)

#         # Use absolute mean SHAP per county
#         svA = np.abs(explainer.shap_values(XA)).mean(axis=0)
#         svB = np.abs(explainer.shap_values(XB)).mean(axis=0)

#         comp_df = pd.DataFrame({
#             "feature": XA.columns,
#             f"{c1}_importance": svA,
#             f"{c2}_importance": svB
#         })
#         comp_df = comp_df.sort_values(f"{c1}_importance", ascending=False).reset_index(drop=True)
#         st.subheader(f"SHAP importances: {c1} vs {c2}")
#         st.dataframe(comp_df)

#         # side-by-side bar chart for top features
#         top_n = st.slider("Top N features to compare", min_value=3, max_value=min(20, len(comp_df)), value=8)
#         top_df = comp_df.head(top_n).set_index("feature")
#         fig, ax = plt.subplots(figsize=(10, 4))
#         top_df.plot.bar(ax=ax)
#         ax.set_ylabel("mean(|SHAP|)")
#         ax.set_title(f"Top {top_n} SHAP importances: {c1} vs {c2}")
#         st.pyplot(fig)

#         # Narrative
#         topA = comp_df.sort_values(f"{c1}_importance", ascending=False).iloc[0]
#         topB = comp_df.sort_values(f"{c2}_importance", ascending=False).iloc[0]
#         st.subheader("📝 County Comparison Summary")
#         st.markdown(f"- **{c1}** most influenced by **{topA['feature']}** (mean|SHAP|={topA[f'{c1}_importance']:.4f}).")
#         st.markdown(f"- **{c2}** most influenced by **{topB['feature']}** (mean|SHAP|={topB[f'{c2}_importance']:.4f}).")
#         st.markdown(
#             "- Interpretation: if a county shows higher NDVI importance it suggests vegetation drive; if lag-features dominate, the county's yields are more influenced by past seasons."
#         )

# # -------------------------
# # PAGE: Storage Planner (separate)
# # -------------------------
# elif page == "Storage Planner":
#     st.header("Storage Planner")
#     st.markdown("Enter the production (tonnes) you want to plan storage for, or use the last forecasted production from Forecasting page.")

#     prod_manual = st.number_input("Production (tonnes)", min_value=0.0, value=0.0, step=1.0)
#     use_last = False
#     if "forecast_df" in locals() and not forecast_df.empty:
#         use_last = st.checkbox("Use final forecasted production from last Forecasting run", value=False)
#         if use_last:
#             prod_manual = float(forecast_df.iloc[-1]["predicted_production"])

#     if prod_manual <= 0:
#         st.info("Enter a production tonnage (or run a forecast and check 'Use final forecast...').")
#     else:
#         storage_plan = pack_storage(prod_manual)
#         st.subheader("Storage Plan")
#         st.json(storage_plan)

# # -------------------------
# # PAGE: Download
# # -------------------------
# elif page == "Download":
#     st.header("Download artifacts")
#     col1, col2 = st.columns(2)
#     with col1:
#         # if local file exists, expose path; else guide to upload
#         if Path(DEFAULT_CSV).exists():
#             st.markdown(f"- Cleaned CSV (default): `{DEFAULT_CSV}`")
#             st.download_button("Download cleaned CSV", data=open(DEFAULT_CSV, "rb").read(), file_name="ndvi_optionb_cleaned.csv", mime="text/csv")
#         else:
#             st.info("Place `ndvi_optionb_cleaned.csv` in /mnt/data to enable direct download from server.")
#     with col2:
#         if Path(DEFAULT_MODEL).exists():
#             st.markdown(f"- Model (default): `{DEFAULT_MODEL}`")
#             st.download_button("Download model (joblib)", data=open(DEFAULT_MODEL, "rb").read(), file_name="rf_optionb_ndvi_model_v2.joblib", mime="application/octet-stream")
#         else:
#             st.info("Place trained joblib model in /mnt/data to enable direct download.")

# st.markdown("---")
# st.caption("App v3 — NDVI Forecasting + SHAP Explainability. For LIME or further features ask to include them in a following update.")
