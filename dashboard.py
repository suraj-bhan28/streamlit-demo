# to run streamlit 
#pip install streamlit pandas scikit-learn joblib
#1. file path 
#2. streamit run filename.py
import pandas as pd

df = pd.read_csv("robot_maintenancecsv.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

import streamlit as st
st.title("Robot Health Dashboard")
st.line_chart(df.set_index('timestamp')['battery_percent'])
st.bar_chart(df['error_code'].value_counts())
st.write("Latest CPU Temp:", df['cpu_temp_c'].iloc[-1])
