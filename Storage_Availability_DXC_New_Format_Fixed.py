import streamlit as st
import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

st.set_page_config(
    page_title="DXC | Storage Availability Prediction",
    page_icon="💾",
    layout="wide"
)


# =========================================================
# DXC REFACE-INSPIRED FRONTEND THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --reface-navy: #10101e;
    --reface-navy-2: #181827;
    --dxc-blue: #0072ce;
    --dxc-cyan: #35bce9;
    --dxc-orange: #ff8757;
    --page-bg: #f7f4f1;
    --ink: #050505;
    --muted: #55545a;
}
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.stApp {
    background: var(--page-bg);
    color: var(--ink);
}
[data-testid="stHeader"] {
    background: rgba(246,244,241,0.96);
}
.block-container {
    max-width: 1600px;
    padding-top: 0.55rem;
    padding-bottom: 2.5rem;
}
section[data-testid="stSidebar"] {
    background: var(--reface-navy);
    border-right: 1px solid #2d3044;
}
section[data-testid="stSidebar"] * {
    color: #f7f7fb;
}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #202238;
    border: 1px dashed #6d7897;
}
.reface-topbar {
    background: var(--reface-navy);
    color: white;
    padding: 17px 34px;
    border-radius: 22px 22px 0 0;
    display: flex;
    align-items: center;
    gap: 24px;
    margin: 0 0 0 0;
    box-shadow: 0 4px 14px rgba(16,17,31,.16);
}
.dxc-mark {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: 3px;
    background: linear-gradient(90deg, #ff9b52, #f6f6fb 48%, #27b7e8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}
.reface-brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -1.2px;
    color: #fff !important;
}
.reface-nav {
    display: flex;
    align-items: center;
    gap: 26px;
    margin-left: 20px;
    font-size: 15px;
    font-weight: 600;
    color: #f3f3f7;
}
.reface-nav span:first-child {
    border-bottom: 3px solid white;
    padding: 0 0 7px 0;
}
.reface-internal {
    margin-left: auto;
    color: #f3f3f7;
    font-size: 13px;
}
.reface-hero {
    position: relative;
    overflow: hidden;
    min-height: 275px;
    padding: 72px 44px 52px 44px;
    background:
      linear-gradient(90deg, rgba(20,25,38,.63), rgba(20,25,38,.23)),
      radial-gradient(ellipse at 72% 28%, rgba(128,151,171,.72), transparent 24%),
      radial-gradient(ellipse at 83% 72%, rgba(67,83,103,.58), transparent 32%),
      linear-gradient(110deg, #9ba3a8 0%, #6e7883 48%, #373d49 100%);
    color: white;
    border-radius: 0;
    margin-bottom: 30px;
}
.reface-hero:before, .reface-hero:after {
    content: "";
    position: absolute;
    width: 62%;
    height: 130%;
    right: -7%;
    top: -37%;
    border: 2px solid rgba(255,255,255,.30);
    border-radius: 50%;
    transform: rotate(-17deg);
}
.reface-hero:after {
    width: 48%;
    right: -3%;
    top: -20%;
    border-color: rgba(255,255,255,.22);
}
.reface-eyebrow {
    color: #f7f7fa !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: 700;
    position: relative;
    z-index: 1;
}
.reface-hero h1 {
    color: white !important;
    font-size: clamp(29px, 3.2vw, 45px);
    font-weight: 800;
    letter-spacing: -1.8px;
    margin: 12px 0 10px 0;
    line-height: 1.1;
    position: relative;
    z-index: 1;
}
.reface-hero p {
    color: #e4e8ef !important;
    font-size: 16px;
    max-width: 780px;
    line-height: 1.65;
    position: relative;
    z-index: 1;
}
.reface-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
    position: relative;
    z-index: 1;
}
.reface-pill {
    display: inline-block;
    border: 1px solid rgba(255,255,255,.30);
    background: rgba(255,255,255,.08);
    padding: 7px 12px;
    border-radius: 20px;
    color: white !important;
    font-size: 12px;
}
.reface-section-title {
    font-size: 30px;
    font-weight: 800;
    color: #050505;
    letter-spacing: -1.25px;
    margin: 10px 0 4px 0;
}
.reface-subtitle {
    color: #606775;
    font-size: 14px;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e1e3e8;
    border-top: 3px solid var(--dxc-blue);
    border-radius: 12px;
    padding: 15px 17px;
    box-shadow: 0 3px 10px rgba(17,24,39,.04);
}
div[data-testid="stMetricLabel"] {
    color: #596170;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: #111827;
    font-weight: 800;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 8px;
    border: 1px solid var(--dxc-blue);
    font-weight: 600;
}
.stButton > button[kind="primary"] {
    background: var(--dxc-blue);
    color: white;
}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border: 1px solid #e1e3e8;
    border-radius: 10px;
    overflow: hidden;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: #eceef2;
    border-radius: 7px 7px 0 0;
    padding: 10px 15px;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    border-bottom: 3px solid var(--dxc-blue);
}
.reface-feature-card {
    background: #fff;
    border: 1px solid #e3e5e9;
    border-radius: 14px;
    padding: 19px 20px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(17,24,39,.04);
}
.reface-feature-card h3 {
    margin: 0 0 8px 0;
    color: #111827;
    font-size: 17px;
}
.reface-feature-card p {
    margin: 0;
    color: #626978;
    font-size: 13px;
    line-height: 1.55;
}
.reface-footer {
    background: #10111f;
    color: #d9dce5;
    padding: 15px 20px;
    border-radius: 8px;
    text-align: center;
    font-size: 12px;
    margin-top: 28px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="reface-topbar">
  <div class="dxc-mark">DXC</div>
  <div class="reface-brand">Storage Availability</div>
  <div class="reface-nav">
    <span>Home</span><span>Prediction</span><span>Risk &amp; Capacity</span>
  </div>
  <div class="reface-internal">DXC Internal&nbsp;&nbsp; •••</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CONFIGURATION
# =========================================================
FILE_NAME = "Storage.xlsx"

EXPECTED_COLUMNS = [
    "IM_DXC",
    "Storage",
    "Facility",
    "Storage_Name",
    "Technology",
    "Raw_Capacity_TB",
    "Usable_Capacity_TB",
    "Used_Capacity_TB",
    "Available_Capacity_TB",
    "Available_Percentage",
    "Comments",
    "YTD"
]


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data(file):
    return pd.read_excel(file)


# =========================================================
# DATA PREPARATION
# =========================================================
def prepare_data(df):

    df = df.copy()

    if len(df.columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {len(EXPECTED_COLUMNS)} columns, "
            f"but found {len(df.columns)} columns."
        )

    # Same column naming used in the notebook
    df.columns = EXPECTED_COLUMNS

    # Convert date
    df["YTD"] = pd.to_datetime(
        df["YTD"],
        errors="coerce"
    )

    # Convert numeric columns
    numeric_cols = [
        "Raw_Capacity_TB",
        "Usable_Capacity_TB",
        "Used_Capacity_TB",
        "Available_Capacity_TB",
        "Available_Percentage"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Replace infinity
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    required_cols = [
        "Storage_Name",
        "YTD",
        "Used_Capacity_TB",
        "Usable_Capacity_TB",
        "Available_Capacity_TB",
        "Available_Percentage"
    ]

    before_rows = len(df)

    df = df.dropna(
        subset=required_cols
    )

    after_rows = len(df)

    # Same sorting used by notebook
    df = df.sort_values(
        ["Storage_Name", "YTD"]
    ).reset_index(drop=True)

    return df, before_rows, after_rows


# =========================================================
# AVAILABILITY / UTILIZATION
# =========================================================
def create_availability_features(df):

    df = df.copy()

    if df["Available_Percentage"].max() > 1:

        df["Available_Pct"] = (
            df["Available_Percentage"] / 100
        )

        conversion_type = (
            "Percentage converted from 0-100 to 0-1"
        )

    else:

        df["Available_Pct"] = (
            df["Available_Percentage"]
        )

        conversion_type = (
            "Percentage already in 0-1 format"
        )

    df["Available_Pct"] = (
        df["Available_Pct"].clip(0, 1)
    )

    df["Utilization_Pct"] = np.where(
        df["Usable_Capacity_TB"] > 0,
        df["Used_Capacity_TB"]
        / df["Usable_Capacity_TB"],
        np.nan
    )

    df["Utilization_Pct"] = (
        df["Utilization_Pct"].clip(0, 1)
    )

    return df, conversion_type


# =========================================================
# LAG FEATURES
# =========================================================
def create_lag_features(df):

    df = df.copy()

    group = df.groupby(
        "Storage_Name"
    )

    for lag in [1, 2, 3]:

        df[f"Available_Lag_{lag}"] = (
            group["Available_Pct"].shift(lag)
        )

        df[f"Used_Lag_{lag}"] = (
            group["Used_Capacity_TB"].shift(lag)
        )

        df[f"Available_TB_Lag_{lag}"] = (
            group["Available_Capacity_TB"].shift(lag)
        )

        df[f"Utilization_Lag_{lag}"] = (
            group["Utilization_Pct"].shift(lag)
        )

    return df


# =========================================================
# USAGE CHANGE / GROWTH
# =========================================================
def create_usage_features(df):

    df = df.copy()

    df["Usage_Change_1M"] = (
        df["Used_Capacity_TB"]
        - df["Used_Lag_1"]
    )

    df["Usage_Change_2M"] = (
        df["Used_Lag_1"]
        - df["Used_Lag_2"]
    )

    df["Usage_Change_3M"] = (
        df["Used_Lag_2"]
        - df["Used_Lag_3"]
    )

    for lag in [1, 2, 3]:

        change_col = (
            f"Usage_Change_{lag}M"
        )

        lag_col = (
            f"Used_Lag_{lag}"
        )

        df[f"Usage_Growth_{lag}M"] = np.where(
            df[lag_col] > 0,
            df[change_col] / df[lag_col],
            np.nan
        )

        df[f"Usage_Growth_{lag}M"] = (
            df[f"Usage_Growth_{lag}M"]
            .clip(-5, 5)
        )

    return df


# =========================================================
# AVAILABILITY CHANGE
# =========================================================
def create_availability_change_features(df):

    df = df.copy()

    df["Availability_Change_1M"] = (
        df["Available_Pct"]
        - df["Available_Lag_1"]
    )

    df["Availability_Change_2M"] = (
        df["Available_Lag_1"]
        - df["Available_Lag_2"]
    )

    df["Availability_Change_3M"] = (
        df["Available_Lag_2"]
        - df["Available_Lag_3"]
    )

    return df


# =========================================================
# ROLLING FEATURES
# =========================================================
def create_rolling_features(df):

    df = df.copy()

    grouped = df.groupby(
        "Storage_Name"
    )

    df["Availability_Rolling_3M"] = (
        grouped["Available_Pct"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                3,
                min_periods=2
            )
            .mean()
        )
    )

    df["Availability_Rolling_6M"] = (
        grouped["Available_Pct"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                6,
                min_periods=3
            )
            .mean()
        )
    )

    df["Availability_Std_3M"] = (
        grouped["Available_Pct"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                3,
                min_periods=2
            )
            .std()
        )
    )

    df["Used_Rolling_3M"] = (
        grouped["Used_Capacity_TB"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                3,
                min_periods=2
            )
            .mean()
        )
    )

    df["Used_Rolling_6M"] = (
        grouped["Used_Capacity_TB"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                6,
                min_periods=3
            )
            .mean()
        )
    )

    df["Utilization_Rolling_3M"] = (
        grouped["Utilization_Pct"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                3,
                min_periods=2
            )
            .mean()
        )
    )

    return df


# =========================================================
# CAPACITY / DATE FEATURES
# =========================================================
def create_capacity_date_features(df):

    df = df.copy()

    df["Capacity_Change_1M"] = (
        df.groupby("Storage_Name")
        ["Usable_Capacity_TB"]
        .diff(1)
    )

    df["Capacity_Change_2M"] = (
        df.groupby("Storage_Name")
        ["Usable_Capacity_TB"]
        .diff(2)
    )

    df["Year"] = df["YTD"].dt.year
    df["Month"] = df["YTD"].dt.month

    df["Month_Index"] = (
        (df["YTD"].dt.year
         - df["YTD"].dt.year.min()) * 12
        + df["YTD"].dt.month
    )

    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    return df


# =========================================================
# TARGET
# =========================================================
def create_target(df):

    df = df.copy()

    df["Target_Next_Month_Available"] = (
        df.groupby("Storage_Name")
        ["Available_Pct"]
        .shift(-1)
    )

    return df


# =========================================================
# MODEL DATASET
# =========================================================
def create_model_dataset(df):

    required_features = [
        "Available_Lag_1",
        "Available_Lag_2",
        "Available_Lag_3",
        "Used_Lag_1",
        "Used_Lag_2",
        "Used_Lag_3",
        "Target_Next_Month_Available"
    ]

    before_rows = len(df)

    df_model = df.dropna(
        subset=required_features
    ).copy()

    after_rows = len(df_model)

    # Save identifiers before encoding
    identifier_data = df_model[
        [
            "Storage_Name",
            "Technology",
            "Facility",
            "YTD"
        ]
    ].copy()

    categorical_columns = [
        "Storage_Name",
        "Technology",
        "Facility"
    ]

    df_model = pd.get_dummies(
        df_model,
        columns=categorical_columns,
        drop_first=False,
        dtype=int
    )

    target_column = (
        "Target_Next_Month_Available"
    )

    exclude_columns = [
        target_column,
        "YTD",
        "IM_DXC",
        "Storage",
        "Comments"
    ]

    feature_columns = [
        col for col in df_model.columns
        if col not in exclude_columns
    ]

    X = df_model[
        feature_columns
    ].copy()

    y = df_model[
        target_column
    ].copy()

    # Replace infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = y.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    df_model = (
        df_model
        .reset_index(drop=True)
    )

    identifier_data = (
        identifier_data
        .reset_index(drop=True)
    )

    # Valid target rows
    valid_rows = y.notna()

    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    df_model = (
        df_model.loc[valid_rows]
        .copy()
    )

    identifier_data = (
        identifier_data.loc[valid_rows]
        .copy()
    )

    # Fill missing features
    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    # Chronological ordering
    sort_index = (
        df_model["YTD"]
        .sort_values()
        .index
    )

    X = (
        X.loc[sort_index]
        .reset_index(drop=True)
    )

    y = (
        y.loc[sort_index]
        .reset_index(drop=True)
    )

    df_model = (
        df_model.loc[sort_index]
        .reset_index(drop=True)
    )

    identifier_data = (
        identifier_data.loc[sort_index]
        .reset_index(drop=True)
    )

    return (
        X,
        y,
        df_model,
        identifier_data,
        feature_columns,
        before_rows,
        after_rows
    )


# =========================================================
# TRAIN MODELS
# =========================================================
def train_models(X, y):

    split_index = int(
        len(X) * 0.80
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    # -----------------------------------------------------
    # XGBoost
    # -----------------------------------------------------
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_test, y_test)
        ],
        verbose=False
    )

    xgb_pred = np.clip(
        xgb_model.predict(X_test),
        0,
        1
    )

    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_pred = np.clip(
        rf_model.predict(X_test),
        0,
        1
    )

    return (
        split_index,
        X_train,
        X_test,
        y_train,
        y_test,
        xgb_model,
        xgb_pred,
        rf_model,
        rf_pred
    )


# =========================================================
# METRICS
# =========================================================
def calculate_metrics(
    y_test,
    y_pred
):

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }


# =========================================================
# LATEST STORAGE SCORING
# =========================================================
def create_latest_scoring_dataset(df, feature_columns, X_reference):
    scoring = df.dropna(subset=[
        "Available_Lag_1", "Available_Lag_2", "Available_Lag_3",
        "Used_Lag_1", "Used_Lag_2", "Used_Lag_3"
    ]).copy()

    if scoring.empty:
        return pd.DataFrame(), pd.DataFrame()

    scoring = (
        scoring.sort_values("YTD")
        .groupby("Storage_Name")
        .tail(1)
        .copy()
    )

    identifiers = scoring[[
        "Storage_Name", "Technology", "Facility", "YTD",
        "Available_Pct", "Utilization_Pct",
        "Usable_Capacity_TB", "Used_Capacity_TB",
        "Available_Capacity_TB"
    ]].copy()

    encoded = pd.get_dummies(
        scoring,
        columns=["Storage_Name", "Technology", "Facility"],
        drop_first=False,
        dtype=int
    )

    encoded = encoded.drop(
        columns=["Target_Next_Month_Available", "YTD", "IM_DXC", "Storage", "Comments"],
        errors="ignore"
    )

    X_latest = encoded.reindex(
        columns=feature_columns,
        fill_value=0
    ).copy()

    X_latest = X_latest.replace([np.inf, -np.inf], np.nan)
    medians = X_reference.median(numeric_only=True)

    for col in X_latest.columns:
        if X_latest[col].isna().any():
            X_latest[col] = X_latest[col].fillna(
                medians[col] if col in medians.index and pd.notna(medians[col]) else 0
            )

    return X_latest.fillna(0), identifiers


# =========================================================
# RISK DASHBOARD
# =========================================================
def create_risk_dashboard(df, latest_identifiers, latest_predictions=None):
    risk_df = latest_identifiers.copy()

    if latest_predictions is not None:
        risk_df["Predicted_Available_%"] = latest_predictions * 100
    else:
        risk_df["Predicted_Available_%"] = np.nan

    risk_df["Current_Availability_%"] = risk_df["Available_Pct"] * 100
    risk_df["Current_Utilization_%"] = risk_df["Utilization_Pct"] * 100

    temp = df.sort_values(["Storage_Name", "YTD"]).copy()
    temp["Utilization_Change_1M"] = (
        temp.groupby("Storage_Name")["Utilization_Pct"].diff()
    )

    recent_change = (
        temp.groupby("Storage_Name")["Utilization_Change_1M"]
        .last()
        .rename("Utilization_Change_1M_%")
        .reset_index()
    )
    recent_change["Utilization_Change_1M_%"] *= 100

    risk_df = risk_df.merge(recent_change, on="Storage_Name", how="left")

    def classify(row):
        a = row["Current_Availability_%"]
        u = row["Current_Utilization_%"]
        p = row["Predicted_Available_%"]
        g = row["Utilization_Change_1M_%"]
        if a <= 10 or u >= 90 or (pd.notna(p) and p <= 10):
            return "Critical"
        if a <= 20 or u >= 80 or (pd.notna(p) and p <= 20) or (pd.notna(g) and g >= 5):
            return "Warning"
        return "Healthy"

    def reason(row):
        reasons = []
        a = row["Current_Availability_%"]
        u = row["Current_Utilization_%"]
        p = row["Predicted_Available_%"]
        g = row["Utilization_Change_1M_%"]
        if a <= 10: reasons.append("Current availability below 10%")
        elif a <= 20: reasons.append("Current availability below 20%")
        if u >= 90: reasons.append("Utilization above 90%")
        elif u >= 80: reasons.append("Utilization above 80%")
        if pd.notna(p):
            if p <= 10: reasons.append("Predicted availability below 10%")
            elif p <= 20: reasons.append("Predicted availability below 20%")
        if pd.notna(g) and g >= 5: reasons.append("Rapid utilization increase")
        return "; ".join(reasons) if reasons else "No immediate risk detected"

    risk_df["Risk_Level"] = risk_df.apply(classify, axis=1)
    risk_df["Risk_Reason"] = risk_df.apply(reason, axis=1)
    return risk_df


# =========================================================
# CAPACITY EXHAUSTION PREDICTION
# =========================================================
def calculate_capacity_exhaustion(df):
    """Estimate months and dates to 80%, 90%, and 95% utilization safely."""
    results = []
    thresholds = {"80": 0.80, "90": 0.90, "95": 0.95}

    for storage_name, group in df.groupby("Storage_Name"):
        group = group.copy()

        # Normalize dates before sorting. Excel date values may arrive as strings.
        group["YTD"] = pd.to_datetime(group["YTD"], errors="coerce")
        group = group.dropna(subset=["YTD"]).sort_values("YTD")
        if len(group) < 3:
            continue

        latest = group.iloc[-1]
        usable = pd.to_numeric(latest["Usable_Capacity_TB"], errors="coerce")
        current_used = pd.to_numeric(latest["Used_Capacity_TB"], errors="coerce")
        current_date = pd.to_datetime(latest["YTD"], errors="coerce")

        if pd.isna(usable) or usable <= 0 or pd.isna(current_used) or pd.isna(current_date):
            continue

        current_util = current_used / usable
        group["Used_Capacity_TB"] = pd.to_numeric(
            group["Used_Capacity_TB"], errors="coerce"
        )
        group["Usage_Change"] = group["Used_Capacity_TB"].diff()
        monthly_growth = group["Usage_Change"].dropna().tail(6).mean()

        row = {
            "Storage_Name": storage_name,
            "Facility": latest["Facility"],
            "Technology": latest["Technology"],
            "Current_Date": pd.Timestamp(current_date),
            "Usable_Capacity_TB": usable,
            "Used_Capacity_TB": current_used,
            "Available_Capacity_TB": latest["Available_Capacity_TB"],
            "Current_Utilization_%": current_util * 100,
            "Monthly_Usage_Growth_TB": monthly_growth,
        }

        for label, threshold in thresholds.items():
            target_used = usable * threshold
            additional = target_used - current_used

            if additional <= 0:
                months = 0.0
            elif pd.isna(monthly_growth) or monthly_growth <= 0:
                months = np.nan
            else:
                months = additional / monthly_growth

            row[f"Months_to_{label}%"] = months
            exhaustion_date = pd.NaT

            # Avoid invalid timestamps and pandas overflow for extreme forecasts.
            if (
                not pd.isna(current_date)
                and not pd.isna(months)
                and np.isfinite(months)
                and 0 <= months <= 1200
            ):
                try:
                    exhaustion_date = (
                        pd.Timestamp(current_date)
                        + pd.DateOffset(months=int(np.ceil(months)))
                    )
                except (OverflowError, ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                    exhaustion_date = pd.NaT

            row[f"Exhaustion_{label}%_Date"] = exhaustion_date

        results.append(row)

    return pd.DataFrame(results)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("DXC | Storage Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Storage Excel file",
    type=["xlsx", "xls"]
)

model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    [
        "XGBoost",
        "Random Forest"
    ]
)

rows_to_display = st.sidebar.slider(
    "Rows to display",
    min_value=5,
    max_value=100,
    value=10
)

show_raw_data = st.sidebar.checkbox(
    "Show Raw Data",
    value=False
)

show_engineered_data = st.sidebar.checkbox(
    "Show Engineered Features",
    value=False
)


# =========================================================
# REFACE-INSPIRED HOME / HERO
# =========================================================
st.markdown("""
<div class="reface-hero">
  <div class="reface-eyebrow">DXC Internal • Storage Intelligence</div>
  <h1>Storage Availability Prediction</h1>
  <p>
    Predict next-month storage availability, identify systems at risk,
    and plan capacity expansion using historical capacity and utilization trends.
  </p>
  <div class="reface-pill-row">
    <span class="reface-pill">XGBoost</span>
    <span class="reface-pill">Random Forest</span>
    <span class="reface-pill">Risk Monitoring</span>
    <span class="reface-pill">Capacity Forecasting</span>
  </div>
</div>
<div class="reface-section-title">Welcome to Storage Insights</div>
<div class="reface-subtitle">
  An integrated workspace for availability prediction, risk prioritization,
  model evaluation and capacity planning.
</div>
<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0 26px 0;">
  <div class="reface-feature-card">
    <h3>01 &nbsp; Availability Prediction</h3>
    <p>Compare XGBoost and Random Forest predictions, review evaluation metrics, and inspect actual-versus-predicted availability.</p>
  </div>
  <div class="reface-feature-card">
    <h3>02 &nbsp; Risk Dashboard</h3>
    <p>Prioritize Critical, Warning and Healthy storage systems with risk reasons and rapid-utilization-growth indicators.</p>
  </div>
  <div class="reface-feature-card">
    <h3>03 &nbsp; Capacity Planning</h3>
    <p>Estimate 80%, 90% and 95% utilization thresholds and review storage systems approaching capacity limits.</p>
  </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# FILE CHECK
# =========================================================
if uploaded_file is None:

    st.info(
        f"Please upload **{FILE_NAME}** using the sidebar."
    )

    st.markdown(
        """
        **Expected columns:**

        1. IM/DXC  
        2. Storage  
        3. Facility  
        4. Storage Name  
        5. Technology  
        6. Raw Capacity (TB)  
        7. Usable Capacity (TB)  
        8. Used Capacity (TB)  
        9. Available Capacity (TB)  
        10. Available Percentage  
        11. Comments  
        12. YTD
        """
    )

    st.stop()


# =========================================================
# EXECUTION
# =========================================================
try:

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------
    raw_df = load_data(
        uploaded_file
    )

    # -----------------------------------------------------
    # Cleaning
    # -----------------------------------------------------
    df, before_cleaning, after_cleaning = (
        prepare_data(raw_df)
    )

    # -----------------------------------------------------
    # Availability
    # -----------------------------------------------------
    df, conversion_type = (
        create_availability_features(df)
    )

    # -----------------------------------------------------
    # Lag
    # -----------------------------------------------------
    df = create_lag_features(df)

    # -----------------------------------------------------
    # Usage
    # -----------------------------------------------------
    df = create_usage_features(df)

    # -----------------------------------------------------
    # Availability changes
    # -----------------------------------------------------
    df = (
        create_availability_change_features(df)
    )

    # -----------------------------------------------------
    # Rolling
    # -----------------------------------------------------
    df = create_rolling_features(df)

    # -----------------------------------------------------
    # Capacity / Date
    # -----------------------------------------------------
    df = create_capacity_date_features(df)

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------
    df = create_target(df)

    # -----------------------------------------------------
    # Model dataset
    # -----------------------------------------------------
    (
        X,
        y,
        df_model,
        identifier_data,
        feature_columns,
        before_model_rows,
        after_model_rows
    ) = create_model_dataset(df)

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    (
        split_index,
        X_train,
        X_test,
        y_train,
        y_test,
        xgb_model,
        xgb_pred,
        rf_model,
        rf_pred
    ) = train_models(
        X,
        y
    )

except Exception as e:

    st.error(
        f"Error while processing the dataset: {e}"
    )

    st.stop()


# =========================================================
# SUCCESS MESSAGE
# =========================================================
st.success(
    "Dataset loaded and model execution completed successfully."
)


# =========================================================
# DATASET OVERVIEW
# =========================================================
st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Original Rows",
    len(raw_df)
)

c2.metric(
    "Cleaned Rows",
    len(df)
)

c3.metric(
    "Rows Removed",
    before_cleaning - after_cleaning
)

c4.metric(
    "Columns",
    len(raw_df.columns)
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Model Rows",
    len(X)
)

c2.metric(
    "Training Rows",
    len(X_train)
)

c3.metric(
    "Testing Rows",
    len(X_test)
)

c4.metric(
    "Features",
    len(feature_columns)
)

st.info(
    f"Availability format: {conversion_type}"
)


# =========================================================
# AVAILABILITY SUMMARY
# =========================================================
st.header("📈 Storage Availability Summary")

avg_availability = (
    df["Available_Pct"].mean()
    * 100
)

avg_utilization = (
    df["Utilization_Pct"].mean()
    * 100
)

max_availability = (
    df["Available_Pct"].max()
    * 100
)

min_availability = (
    df["Available_Pct"].min()
    * 100
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Average Availability",
    f"{avg_availability:.2f}%"
)

c2.metric(
    "Average Utilization",
    f"{avg_utilization:.2f}%"
)

c3.metric(
    "Maximum Availability",
    f"{max_availability:.2f}%"
)

c4.metric(
    "Minimum Availability",
    f"{min_availability:.2f}%"
)


# =========================================================
# STORAGE SUMMARY
# =========================================================
st.subheader("Storage Summary")

summary_df = df[
    [
        "Storage_Name",
        "YTD",
        "Used_Capacity_TB",
        "Available_Capacity_TB",
        "Available_Pct",
        "Utilization_Pct"
    ]
].copy()

summary_df[
    "Available_Pct"
] *= 100

summary_df[
    "Utilization_Pct"
] *= 100

st.dataframe(
    summary_df.head(rows_to_display),
    use_container_width=True
)


# =========================================================
# RAW DATA
# =========================================================
if show_raw_data:

    st.header("📋 Raw Storage Data")

    st.dataframe(
        raw_df,
        use_container_width=True
    )


# =========================================================
# ENGINEERED DATA
# =========================================================
if show_engineered_data:

    st.header("🔧 Engineered Features")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Lag Features",
            "Usage Features",
            "Rolling Features",
            "Capacity & Date"
        ]
    )

    with tab1:

        columns = [
            "Storage_Name",
            "YTD",
            "Available_Lag_1",
            "Available_Lag_2",
            "Available_Lag_3",
            "Used_Lag_1",
            "Used_Lag_2",
            "Used_Lag_3"
        ]

        st.dataframe(
            df[columns].head(
                rows_to_display
            ),
            use_container_width=True
        )

    with tab2:

        columns = [
            "Storage_Name",
            "YTD",
            "Usage_Change_1M",
            "Usage_Growth_1M",
            "Usage_Change_2M",
            "Usage_Growth_2M",
            "Usage_Change_3M",
            "Usage_Growth_3M"
        ]

        st.dataframe(
            df[columns].head(
                rows_to_display
            ),
            use_container_width=True
        )

    with tab3:

        columns = [
            "Storage_Name",
            "YTD",
            "Availability_Rolling_3M",
            "Availability_Rolling_6M",
            "Availability_Std_3M",
            "Used_Rolling_3M",
            "Used_Rolling_6M",
            "Utilization_Rolling_3M"
        ]

        st.dataframe(
            df[columns].head(
                rows_to_display
            ),
            use_container_width=True
        )

    with tab4:

        columns = [
            "Storage_Name",
            "YTD",
            "Usable_Capacity_TB",
            "Capacity_Change_1M",
            "Capacity_Change_2M",
            "Year",
            "Month",
            "Month_Index",
            "Month_Sin",
            "Month_Cos"
        ]

        st.dataframe(
            df[columns].head(
                rows_to_display
            ),
            use_container_width=True
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
xgb_metrics = calculate_metrics(
    y_test,
    xgb_pred
)

rf_metrics = calculate_metrics(
    y_test,
    rf_pred
)

st.header("🤖 Model Performance")

performance_df = pd.DataFrame(
    {
        "Metric": [
            "R² Score",
            "R² Percentage",
            "MAE",
            "RMSE"
        ],

        "XGBoost": [
            xgb_metrics["r2"],
            xgb_metrics["r2"] * 100,
            xgb_metrics["mae"] * 100,
            xgb_metrics["rmse"] * 100
        ],

        "Random Forest": [
            rf_metrics["r2"],
            rf_metrics["r2"] * 100,
            rf_metrics["mae"] * 100,
            rf_metrics["rmse"] * 100
        ]
    }
)

st.dataframe(
    performance_df,
    use_container_width=True
)


# =========================================================
# SELECTED MODEL
# =========================================================
if model_choice == "XGBoost":

    model = xgb_model
    y_pred = xgb_pred
    metrics = xgb_metrics

else:

    model = rf_model
    y_pred = rf_pred
    metrics = rf_metrics


c1, c2, c3 = st.columns(3)

c1.metric(
    "R² Score",
    f"{metrics['r2']:.4f}"
)

c2.metric(
    "MAE",
    f"{metrics['mae'] * 100:.2f} percentage points"
)

c3.metric(
    "RMSE",
    f"{metrics['rmse'] * 100:.2f} percentage points"
)


# =========================================================
# TRAIN / TEST PERIOD
# =========================================================
st.subheader("Training and Testing Period")

period_c1, period_c2 = st.columns(2)

period_c1.write(
    f"**Training:** "
    f"{df_model['YTD'].iloc[0].date()} "
    f"to "
    f"{df_model['YTD'].iloc[split_index - 1].date()}"
)

period_c2.write(
    f"**Testing:** "
    f"{df_model['YTD'].iloc[split_index].date()} "
    f"to "
    f"{df_model['YTD'].iloc[-1].date()}"
)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================
st.header(
    f"🎯 {model_choice} - Actual vs Predicted"
)

results = identifier_data.iloc[
    split_index:
].copy().reset_index(drop=True)

results["Actual_Available_%"] = (
    y_test.values * 100
)

results["Predicted_Available_%"] = (
    y_pred * 100
)

results["Error_%"] = (
    results["Actual_Available_%"]
    - results["Predicted_Available_%"]
)

results["Absolute_Error_%"] = (
    results["Error_%"].abs()
)

st.dataframe(
    results,
    use_container_width=True
)


# =========================================================
# ACTUAL VS PREDICTED CHART
# =========================================================
st.subheader(
    "Actual vs Predicted Availability"
)

chart_df = results[
    [
        "YTD",
        "Actual_Available_%",
        "Predicted_Available_%"
    ]
].copy()

chart_df = chart_df.set_index(
    "YTD"
)

st.line_chart(
    chart_df
)


# =========================================================
# PREPARE LATEST STORAGE SCORING FOR RISK DASHBOARD
# =========================================================
st.header("🚨 Storage Risk Dashboard")

# Build one latest record per storage system and generate a
# next-availability prediction using the trained XGBoost model.
latest_X, latest_identifiers = create_latest_scoring_dataset(
    df,
    feature_columns,
    X
)

latest_pred = None

if not latest_X.empty and not latest_identifiers.empty:
    try:
        latest_pred = xgb_model.predict(latest_X)
    except Exception:
        latest_pred = None

# Fallback: if lag history is insufficient for prediction,
# still show the risk dashboard using the latest actual metrics.
if latest_identifiers.empty:
    latest_identifiers = (
        df.sort_values("YTD")
        .groupby("Storage_Name")
        .tail(1)[[
            "Storage_Name", "Technology", "Facility", "YTD",
            "Available_Pct", "Utilization_Pct", "Usable_Capacity_TB",
            "Used_Capacity_TB", "Available_Capacity_TB"
        ]].copy()
    )

risk_dashboard = create_risk_dashboard(
    df,
    latest_identifiers,
    latest_pred
)

critical_count = int((risk_dashboard["Risk_Level"] == "Critical").sum())
warning_count = int((risk_dashboard["Risk_Level"] == "Warning").sum())
healthy_count = int((risk_dashboard["Risk_Level"] == "Healthy").sum())

c1, c2, c3 = st.columns(3)
c1.metric("🔴 Critical", critical_count)
c2.metric("🟡 Warning", warning_count)
c3.metric("🟢 Healthy", healthy_count)

risk_summary = pd.DataFrame({
    "Risk Level": ["Critical", "Warning", "Healthy"],
    "Storage Count": [critical_count, warning_count, healthy_count]
})

st.subheader("Risk Distribution")
st.bar_chart(risk_summary.set_index("Risk Level"))

risk_filter = st.selectbox(
    "Filter Risk Level",
    ["All", "Critical", "Warning", "Healthy"]
)

risk_display = risk_dashboard.copy()
if risk_filter != "All":
    risk_display = risk_display[risk_display["Risk_Level"] == risk_filter]

risk_display = risk_display[[
    "Storage_Name", "Facility", "Technology", "YTD",
    "Current_Availability_%", "Current_Utilization_%",
    "Predicted_Available_%", "Utilization_Change_1M_%",
    "Risk_Level", "Risk_Reason"
]].sort_values(
    ["Risk_Level", "Current_Availability_%"],
    ascending=[True, True]
)

st.subheader("Storage Risk Details")
st.dataframe(risk_display, use_container_width=True, hide_index=True)


# =========================================================
# CAPACITY EXHAUSTION PREDICTION
# =========================================================
st.header("📅 Capacity Exhaustion Prediction")
st.markdown(
    "This planning estimate uses recent historical monthly growth in "
    "used capacity to estimate when storage may reach 80%, 90% and 95% utilization. "
    "If usage is flat or declining, a future exhaustion date is not invented."
)

exhaustion_df = calculate_capacity_exhaustion(df)

if exhaustion_df.empty:
    st.warning("Not enough historical data to calculate capacity exhaustion.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Storage Systems", len(exhaustion_df))
    c2.metric("At / Above 80%", int((exhaustion_df["Current_Utilization_%"] >= 80).sum()))
    c3.metric("At / Above 90%", int((exhaustion_df["Current_Utilization_%"] >= 90).sum()))
    c4.metric("At / Above 95%", int((exhaustion_df["Current_Utilization_%"] >= 95).sum()))

    selected_storage = st.selectbox(
        "Select Storage for Capacity Analysis",
        sorted(exhaustion_df["Storage_Name"].unique())
    )

    selected = exhaustion_df[
        exhaustion_df["Storage_Name"] == selected_storage
    ].iloc[0]

    months_90 = selected["Months_to_90%"]
    date_90 = selected["Exhaustion_90%_Date"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Utilization", f"{selected['Current_Utilization_%']:.1f}%")
    c2.metric(
        "Monthly Usage Growth",
        "N/A" if pd.isna(selected["Monthly_Usage_Growth_TB"])
        else f"{selected['Monthly_Usage_Growth_TB']:.2f} TB"
    )
    c3.metric(
        "Time to 90%",
        "Already reached" if months_90 == 0
        else "No positive growth" if pd.isna(months_90)
        else f"{months_90:.1f} months"
    )
    c4.metric(
        "90% Utilization Date",
        "Already reached" if months_90 == 0
        else "Not available" if pd.isna(date_90)
        else date_90.strftime("%b %Y")
    )

    selected_table = pd.DataFrame({
        "Threshold": ["80%", "90%", "95%"],
        "Months Remaining": [
            selected["Months_to_80%"],
            selected["Months_to_90%"],
            selected["Months_to_95%"]
        ],
        "Estimated Date": [
            selected["Exhaustion_80%_Date"],
            selected["Exhaustion_90%_Date"],
            selected["Exhaustion_95%_Date"]
        ]
    })
    selected_table["Months Remaining"] = selected_table["Months Remaining"].round(1)
    selected_table["Estimated Date"] = pd.to_datetime(
        selected_table["Estimated Date"], errors="coerce"
    ).dt.strftime("%b %Y").fillna("Not available")

    st.subheader(f"Capacity Forecast - {selected_storage}")
    st.dataframe(selected_table, use_container_width=True, hide_index=True)

    st.subheader("All Storage Capacity Forecasts")
    exhaustion_display = exhaustion_df.copy()

    for col in [
        "Usable_Capacity_TB", "Used_Capacity_TB", "Available_Capacity_TB",
        "Current_Utilization_%", "Monthly_Usage_Growth_TB",
        "Months_to_80%", "Months_to_90%", "Months_to_95%"
    ]:
        exhaustion_display[col] = exhaustion_display[col].round(2)

    st.dataframe(exhaustion_display, use_container_width=True, hide_index=True)

    st.subheader("⚠️ Nearest 90% Capacity Threshold")
    nearest = exhaustion_df[
        exhaustion_df["Months_to_90%"].notna()
    ].sort_values("Months_to_90%").head(10)

    if nearest.empty:
        st.info("No storage systems currently have a positive usage-growth trend.")
    else:
        st.dataframe(nearest, use_container_width=True, hide_index=True)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
st.header("🔍 Feature Importance")

importance = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "Importance",
    ascending=False
).reset_index(
    drop=True
)

st.dataframe(
    importance.head(20),
    use_container_width=True
)

st.subheader(
    "Top 15 Features"
)

top_features = (
    importance.head(15)
    .set_index("Feature")
)

st.bar_chart(
    top_features["Importance"]
)


# =========================================================
# PREDICTION SUMMARY
# =========================================================
st.header("📌 Prediction Summary")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Average Actual Availability",
    f"{y_test.mean() * 100:.2f}%"
)

c2.metric(
    "Average Predicted Availability",
    f"{y_pred.mean() * 100:.2f}%"
)

c3.metric(
    "Average Absolute Error",
    f"{np.mean(np.abs(y_test.values - y_pred)) * 100:.2f} pp"
)


# =========================================================
# LOW AVAILABILITY PREDICTIONS
# =========================================================
st.header("⚠️ Low Availability Predictions")

threshold = st.slider(
    "Show predictions below availability (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)

low_availability = results[
    results["Predicted_Available_%"]
    <= threshold
].copy()

st.write(
    f"Predictions below {threshold:.0f}%: "
    f"**{len(low_availability)}**"
)

if len(low_availability) > 0:

    st.dataframe(
        low_availability,
        use_container_width=True
    )

else:

    st.success(
        "No predictions are below the selected threshold."
    )


# =========================================================
# DOWNLOAD
# =========================================================
csv_data = results.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Prediction Results",
    data=csv_data,
    file_name=(
        "storage_availability_predictions.csv"
    ),
    mime="text/csv"
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================
with st.expander(
    "🔧 Model Configuration"
):

    if model_choice == "XGBoost":

        st.write(
            {
                "Model": "XGBRegressor",
                "n_estimators": 1000,
                "learning_rate": 0.03,
                "max_depth": 5,
                "min_child_weight": 3,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "random_state": 42,
                "n_jobs": -1
            }
        )

    else:

        st.write(
            {
                "Model": "RandomForestRegressor",
                "n_estimators": 500,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1
            }
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown(
    '<div class="reface-footer">DXC Internal &nbsp; | &nbsp; Storage Availability Prediction '
    '&nbsp; | &nbsp; XGBoost / Random Forest &nbsp; | &nbsp; Risk &amp; Capacity Insights</div>',
    unsafe_allow_html=True
)
