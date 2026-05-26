import streamlit as st
import pandas as pd

st.set_page_config(page_title="UFC CLV Dashboard", layout="wide")

st.title("UFC CLV Dashboard")

clv = pd.read_csv("ufc_clv_results.csv")
closing = pd.read_csv("ufc_closing_lines.csv")
latest = pd.read_csv("ufc_latest_market_snapshot.csv")

st.header("CLV Results")
st.dataframe(clv, use_container_width=True)

st.header("Closing Lines")
st.dataframe(closing, use_container_width=True)

st.header("Latest Market Snapshot")
st.dataframe(latest, use_container_width=True)
