import streamlit as st
import pandas as pd
import joblib
from utils.encoder import encode_role,encode_format

import os

model_path = os.path.join("model", "selection_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Cricket Selection AI", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    .stSelectbox, .stNumberInput, .stSlider {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 12px;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.5);
    }
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 20px;
        text-align: center;
    }
    h1 {
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏏 Premium Cricket Squad AI")
st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    role = st.selectbox("Select Player Role", ["Batsman", "Bowler", "Wicketkeeper", "All-Rounder"])
    format_type = st.selectbox("Match Format", ["T20", "ODI", "Test"])


st.sidebar.header("Player Statistics")

# Commonalities
matches = st.sidebar.number_input("Matches Played", 0, 500)
innings = st.sidebar.number_input("Innings", 0, 500)

batting_stats = {}
bowling_stats = {}

if role in ["Batsman", "All-Rounder", "Wicketkeeper"]:
    st.sidebar.subheader("Batting Stats")
    batting_stats['runs'] = st.sidebar.number_input("Runs", 0, 20000)
    batting_stats['balls'] = st.sidebar.number_input("Balls Faced", 0, 20000)
    batting_stats['avg'] = st.sidebar.slider("Batting Average", 0.0, 100.0)
    batting_stats['sr'] = st.sidebar.slider("Strike Rate", 0.0, 200.0)
    batting_stats['last5_runs'] = st.sidebar.number_input("Last 5 Match Runs", 0, 500)

if role in ["Bowler", "All-Rounder"]:
    st.sidebar.subheader("Bowling Stats")
    bowling_stats['wickets'] = st.sidebar.number_input("Wickets", 0, 1000)
    bowling_stats['runs_conceded'] = st.sidebar.number_input("Runs Conceded", 0, 20000)
    bowling_stats['bowl_avg'] = st.sidebar.slider("Bowling Average", 0.0, 100.0)
    bowling_stats['bowl_sr'] = st.sidebar.slider("Bowling Strike Rate", 0.0, 100.0)
    bowling_stats['econ'] = st.sidebar.slider("Economy Rate", 0.0, 20.0)
    bowling_stats['best_bowling'] = st.sidebar.text_input("Best Bowling (e.g. 5/20)", "0/0")


if st.button("Predict Selection"):
    role_code = encode_role(role)
    format_code = encode_format(format_type)

    # Use values from the stats dictionaries, defaulting to 0 if not present
    runs = batting_stats.get('runs', 0)
    balls = batting_stats.get('balls', 0)
    runs_conceded = bowling_stats.get('runs_conceded', 0)
    last5_runs = batting_stats.get('last5_runs', 0)
    batting_avg = batting_stats.get('avg', 0.0)
    strike_rate = batting_stats.get('sr', 0.0)
    
    # New features
    wickets = bowling_stats.get('wickets', 0)
    bowl_avg = bowling_stats.get('bowl_avg', 0.0)
    bowl_sr = bowling_stats.get('bowl_sr', 0.0)
    econ = bowling_stats.get('econ', 0.0)
    
    # Existing model uses last5_sr which was a slider. Let's default to a plausible value or 0.
    last5_sr = 120.0 if role in ["Batsman", "All-Rounder"] else 0.0

    data = pd.DataFrame([[
        role_code,
        format_code,
        matches,
        innings,
        runs,
        balls,
        runs_conceded,
        last5_runs,
        last5_sr,
        batting_avg,
        strike_rate,
        wickets,
        bowl_avg,
        bowl_sr,
        econ
    ]], columns=[
        "role",
        "format",
        "matches_played",
        "innings",
        "runs",
        "balls",
        "runs_conceded",
        "last5_runs",
        "last5_strike_rate",
        "batting_average",
        "strike_rate",
        "wickets",
        "bowling_average",
        "bowling_strike_rate",
        "economy_rate"
    ])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]*100

    with col2:
        st.markdown(f"""
        <div class="result-card">
            <h2>Selection Analysis</h2>
            <p style='font-size: 1.2rem;'>The AI model has evaluated the player metrics for the <b>{role}</b> role in <b>{format_type}</b> format.</p>
            <div style='margin: 20px 0;'>
                <h3 style='color: {"#00ff00" if prediction == 1 else "#ff4b4b"}'>
                    {"✅ HIGHLY RECOMMENDED" if prediction == 1 else "❌ NOT RECOMMENDED"}
                </h3>
            </div>
            <p style='font-size: 1.1rem;'>Confidence Level: <b>{probability:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(int(probability))
        
        if prediction == 1:
            st.success("Analysis complete. This player shows strong potential for squad selection.")
        else:
            st.warning("Analysis complete. The player might need more consistent performance in key metrics.")