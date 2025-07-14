# ------------------------------------------------------------
#  app.py  –  Robot Failure 15‑Minute Early‑Warning Dashboard
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1. CONFIG & UTILITIES
# ─────────────────────────────────────────────────────────────
DATA_PATH  = Path("robot_maintenancecsv.csv")       # fallback CSV
MODEL_PATH = Path("robot_failure_predictor.pkl")     # trained model

ROLL_COLS = [
    "battery_percent", "battery_voltage", "cpu_temp_c",
    "motor_current_a", "task_load"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates EXACTLY the feature logic used during training."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time features
    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    # Rolling stats (5‑row = 5‑min window in original data)
    for col in ROLL_COLS:
        df[f"{col}_mean5"] = df[col].rolling(5, min_periods=1).mean()
        df[f"{col}_std5"]  = df[col].rolling(5, min_periods=1).std()

    # Deltas
    for col in ROLL_COLS:
        df[f"{col}_diff1"] = df[col].diff()

    return df.dropna()          # drop early NaNs created by diff/std


# ─────────────────────────────────────────────────────────────
# 2. LOAD MODEL ONCE
# ─────────────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
FEATURE_LIST = list(model.feature_names_in_)         # ground‑truth list


# ─────────────────────────────────────────────────────────────
# 3. STREAMLIT LAYOUT
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Robot Failure Dashboard", layout="wide")
st.title("🤖 Robot Failure Prediction (15‑minute Early Warning)")


# Sidebar: data source
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload sensor CSV", type=["csv"])
    autorefresh   = st.checkbox("Auto‑refresh every 30 s", value=False)

# Choose the dataframe source
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    df_raw = pd.read_csv(DATA_PATH)

# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING  ➜  PREDICTION
# ─────────────────────────────────────────────────────────────
df_feat = engineer_features(df_raw)

# Ensure column names match the model (add missing, reorder, drop extras)
for col in FEATURE_LIST:
    if col not in df_feat:
        df_feat[col] = 0                       # fill missing engineered col
df_feat = df_feat[FEATURE_LIST]               # exact order required

# Predict
df_raw = df_raw.iloc[-len(df_feat):].reset_index(drop=True)   # align rows
df_raw["fail_prob"] = model.predict_proba(df_feat)[:, 1]
df_raw["fail_pred"] = (df_raw["fail_prob"] >= 0.5).astype(int)
latest = df_raw.iloc[-1]                                      # last row


# ─────────────────────────────────────────────────────────────
# 5. KPI Tiles
 #Tiles
# ─────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("🔋 Battery %",        f"{latest['battery_percent']:.1f}%")
k2.metric("🌡️ CPU Temp (°C)",   f"{latest['cpu_temp_c']:.1f}")
k3.metric("⚠️ Fail Prob (15 min)", f"{latest['fail_prob']*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# 6. Charts & Data
# ─────────────────────────────────────────────────────────────
# Convert timestamp once for plotting
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
df_raw = df_raw.set_index("timestamp")

st.subheader("Sensor Trends")
st.line_chart(df_raw[["battery_percent", "cpu_temp_c", "motor_current_a"]])

st.subheader("Failure Probability Over Time")
st.line_chart(df_raw["fail_prob"])

st.subheader("Error‑Code Distribution")
st.bar_chart(df_raw["error_code"].value_counts())

with st.expander("Raw & Predicted Data (last 200 rows)"):
    st.dataframe(df_raw.reset_index().tail(200), use_container_width=True)

# ───────────────────────────────────────────────# ─────────────────
# ─────────────────────────────────────────────────────────────
# 5. KPI──────────────
# 7. Optional Auto‑Refresh
# ─────────────────────────────────────────────────────────────
if autorefresh:
    st.rerun()


