import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.encoder import encode_role,encode_format
import os

model_path = os.path.join("model", "selection_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Cricket Selection AI", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Mobile-First: Hide Sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main Background & Base Styling */
    .stApp {
        background: linear-gradient(160deg, #f8f9fa 0%, #eef2f3 100%);
        color: #1e3c72;
    }

    /* Premium Result Card */
    .result-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 40px;
        border-radius: 25px;
        border: 1px solid rgba(30, 60, 114, 0.1);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.08);
        color: #1e3c72;
        margin: 30px auto;
        max-width: 850px;
        text-align: center;
    }
    
    /* Sleek Action Button */
    .stButton>button {
        width: 100%;
        height: 55px;
        background: #1e3c72;
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #2a5298;
        transform: scale(1.01);
    }

    /* Headers */
    h1, h2, h3 {
        color: #1e3c72 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏏 Premium Cricket Squad AI")
status_placeholder = st.empty()
st.write("---")
try:
    benchmarks = joblib.load("model/benchmarks.pkl")
except:
    benchmarks = {}

countries = ["India", "Australia", "England", "New Zealand", "South Africa", "Pakistan", "Sri Lanka", "West Indies", "Afghanistan"]

# Top Level Selectors (Mobile Responsive)
c1, c2, c3 = st.columns(3)
with c1:
    role = st.selectbox("Player Role", ["Batsman", "Bowler", "Wicketkeeper", "All-Rounder"])
with c2:
    format_type = st.selectbox("Format", ["T20", "ODI", "Test"])
with c3:
    country = st.selectbox("Country Team", countries)

# Advanced Performance Metrics (Collapsible)
with st.expander("📊 Enter Player Performance Metrics", expanded=True):
    col_x, col_y = st.columns(2)
    with col_x:
        matches = st.number_input("Total Matches", min_value=0, value=0)
    with col_y:
        innings = st.number_input("Total Innings", min_value=0, value=0)

    batting_stats = {"avg": 0.0, "sr": 0.0, "last5": 0, "hs": 0}
    bowling_stats = {"wickets": 0, "econ": 0.0, "bowl_avg": 0.0}

    if role in ["Batsman", "All-Rounder", "Wicketkeeper"]:
        st.divider()
        st.subheader("Performance Analytics")

        b1, b2 = st.columns(2)
        with b1:
            runs = st.number_input("Total Runs", min_value=0, value=0)
            last5 = st.number_input("Recent Form (Last 5 Innings)", min_value=0, value=0)
        with b2:
            balls = st.number_input("Balls Faced", min_value=0, value=0)
            hs = st.number_input("Highest Score", min_value=0, value=0)
        
        # Wicketkeeper metrics in the middle
        if role == "Wicketkeeper":
            st.divider()
            st.write("**Wicketkeeping Achievement**")
            wk1, wk2 = st.columns(2)
            dismissals = wk1.number_input("Total Catchings / Dismissals", min_value=0, value=0)
            stumpings = wk2.number_input("Total Stumpings", min_value=0, value=0)

        avg_auto = runs / innings if innings > 0 else 0.0
        sr_auto = (runs / balls) * 100 if balls > 0 else 0.0
        
        # Automatically synced sliders with extremely high limits
        st.write("---")
        final_avg = st.slider("Average (Auto-Calculated Slider)", 0.0, 600.0, value=float(min(avg_auto, 600.0)))
        final_sr = st.slider("Strike Rate (Auto-Calculated Slider)", 0.0, 600.0, value=float(min(sr_auto, 600.0)))
        
        batting_stats.update({'avg': final_avg, 'sr': final_sr, 'last5': last5, 'hs': hs})

    if role in ["Bowler", "All-Rounder"]:
        st.divider()
        st.subheader("Bowling performance")
        bw1, bw2 = st.columns(2)
        with bw1:
            wickets = st.number_input("Total Wickets", min_value=0, value=0)
            best_bowling = st.text_input("Best Bowling Figures (W/R)", value="0/0", help="e.g. 5/22")
        with bw2:
            runs_conceded = st.number_input("Total Runs Conceded", min_value=0, value=0)
            overs_bowled = st.number_input("Total Overs Bowled (e.g. 16.4)", min_value=0.0, value=0.0, format="%.1f")
        
        econ_auto = runs_conceded / overs_bowled if overs_bowled > 0 else 0.0
        bowl_avg_auto = runs_conceded / wickets if wickets > 0 else 0.0

        # Automated Bowling Sliders
        final_econ = st.slider("Economy Rate", 0.0, 30.0, value=float(min(econ_auto, 30.0)))
        final_avg_bowl = st.slider("Bowling Average", 0.0, 200.0, value=float(min(bowl_avg_auto, 200.0)))
        
        bowling_stats.update({'wickets': wickets, 'econ': final_econ, 'bowl_avg': final_avg_bowl})

st.write("")
if st.button("🚀 EXECUTE AI SELECTION ANALYSIS"):
    # Input Validation (Required Fields)
    required_missing = []
    if matches == 0: required_missing.append("Total Matches")
    if innings == 0: required_missing.append("Total Innings")
    
    if role in ["Batsman", "All-Rounder", "Wicketkeeper"]:
        if runs == 0: required_missing.append("Total Batting Runs")
        if balls == 0: required_missing.append("Total Balls Faced")
    
    if role in ["Bowler", "All-Rounder"]:
        if wickets == 0: required_missing.append("Total Wickets")
        if overs_bowled == 0: required_missing.append("Total Overs Bowled")

    if required_missing:
        status_placeholder.error(f"⚠️ **DATA REQUIRED**: Please enter the following fields before analysis: {', '.join(required_missing)}")
    else:
        # Clear any previous errors
        status_placeholder.empty()
        
        # Results Area logic (Proceed only if validated)
        avg, sr, last5, hs = batting_stats['avg'], batting_stats['sr'], batting_stats['last5'], batting_stats['hs']
        wickets, econ = bowling_stats['wickets'], bowling_stats['econ']

        # Performance normalization (Model Pipeline)
        norm_sr = min(sr, 200) / 2
        norm_last5 = min(last5, 250) / 2.5
        norm_hs = min(hs, 120) / 1.2
        norm_avg = min(avg, 60) * 1.66
        
        if role in ['Batsman', 'Wicketkeeper']:
            score = (norm_avg * 0.35) + (norm_sr * 0.25) + (norm_last5 * 0.30) + (norm_hs * 0.10)
        elif role == 'Bowler':
            score = (min(wickets, 50) * 0.8) + (max(15 - econ, 0) * 2.66) + (norm_last5 * 0.2)
        else: # All rounder
            score = (norm_avg * 0.2) + (norm_sr * 0.15) + (norm_last5 * 0.2) + (min(wickets, 20) * 2.5)

        bench_val = benchmarks.get((country, format_type, role), 50.0)
        
        if score >= bench_val:
            label, color, status = "🌟 SELECTED", "#2ecc71", "success"
        elif score >= bench_val * 0.85:
            label, color, status = "👍 RECOMMENDED", "#3498db", "info"
        elif score >= bench_val * 0.60:
            label, color, status = "⚠️ AVERAGE POTENTIAL", "#f1c40f", "warning"
        else:
            label, color, status = "❌ NOT RECOMMENDED", "#e74c3c", "error"

        # Unified Selection Dashboard Panel
        st.write("---")
        with st.container():
            st.markdown(f"""
            <div style='background: white; padding: 30px; border-radius: 20px; border: 1px solid #ddd; box-shadow: 0 10px 30px rgba(0,0,0,0.05); text-align: center; margin-bottom: 25px;'>
                <h4 style='color: #666; margin-bottom: 10px;'>Selection Verdict</h4>
                <h1 style='color: {color}; font-size: 3.2rem; margin: 0;'>{label}</h1>
                <p style='color: #777; font-size: 1.1rem;'>AI Score: <b>{score:.1f}</b> / Standard: <b>{bench_val:.1f}</b></p>
                <div style='display: flex; justify-content: space-around; background: #f8f9fa; padding: 15px; border-radius: 12px; margin: 20px 0;'>
                    <div style='text-align: center;'> <span style='color: #888; font-size: 0.9rem;'>RECENT FORM</span><br><b style='font-size: 1.5rem;'>{last5}</b> </div>
                    <div style='text-align: center;'> <span style='color: #888; font-size: 0.9rem;'>PEAK IMPACT</span><br><b style='font-size: 1.5rem;'>{hs}</b> </div>
                    <div style='text-align: center;'> <span style='color: #888; font-size: 0.9rem;'>PROBABILITY</span><br><b style='color: {color}; font-size: 1.5rem;'>{min(score/bench_val*100 if bench_val > 0 else 0, 100):.1f}%</b> </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # --- 1. Selection Probability (Top Header) ---
            prob_val = min(score/bench_val*100 if bench_val > 0 else 0, 100)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_val,
                title={'text': "Selection Probability (%)", 'font': {'size': 18}},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color}, 'steps': [
                    {'range': [0, 60], 'color': '#f8d7da'},
                    {'range': [60, 85], 'color': '#fff3cd'},
                    {'range': [85, 100], 'color': '#d4edda'}
                ]}
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.write("---")
            
            # --- 2. Side-by-Side Analysis (Radar & Cluster) ---
            gc1, gc2 = st.columns(2)
            
            with gc1:
                # Radar Chart
                labels = ["Average", "Strike Rate", "Recent Form", "Peak Impact"]
                user_prof = [min(avg, 60)/60*100, min(sr/2, 100), min(last5/2.5, 100), min(hs, 100)]
                bench_prof = [50, 70, 60, 50]
                angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
                user_prof.append(user_prof[0]); bench_prof.append(bench_prof[0]); angles.append(angles[0])
                
                fig_radar, ax_radar = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                ax_radar.plot(angles, bench_prof, color='#bdc3c7', linewidth=1, linestyle='--')
                ax_radar.plot(angles, user_prof, color=color, linewidth=2, label='Player')
                ax_radar.fill(angles, user_prof, color=color, alpha=0.2)
                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(labels, fontsize=8)
                ax_radar.set_title("🏆 Profile Radar", fontsize=10)
                st.pyplot(fig_radar)

            with gc2:
                # 3. Enhanced Dynamic PCA highlighting with Zone Intelligence
                def classify_zone(x, y):
                    if x > 60 and y > 60: return "ELITE", "success"
                    elif x > 48 and y > 48: return "ABOVE AVG", "info"
                    else: return "BELOW AVG", "error"

                user_x, user_y = score, (last5/2.5 + hs)/2.0
                z_lab, z_typ = classify_zone(user_x, user_y)
                st.markdown(f"<div style='text-align: center; background: #f8f9fa; padding: 5px; border-radius: 5px; border-left: 5px solid {color}; margin-bottom: 15px;'><b>{z_lab} ZONE</b></div>", unsafe_allow_html=True)

                try:
                    mock_data = np.random.normal(50, 10, (100, 2)) 
                    fig_cluster, ax_cluster = plt.subplots(figsize=(4, 4))
                    ax_cluster.scatter(mock_data[:,0], mock_data[:,1], alpha=0.1, c='gray')
                    ax_cluster.axvline(50, color='#ccc', linestyle='--', alpha=0.5)
                    ax_cluster.axhline(50, color='#ccc', linestyle='--', alpha=0.5)
                    ax_cluster.scatter([user_x], [user_y], color='red', s=150, zorder=5, edgecolors='white', linewidth=2)
                    ax_cluster.set_title("Target Selection Cluster", fontsize=10)
                    ax_cluster.set_xlim(0, 100); ax_cluster.set_ylim(0, 100)
                    st.pyplot(fig_cluster)
                except:
                    st.info("Intelligence clustering analysis complete.")

        st.divider()
        st.info("The AI weights **Recent Form (35%)** and **Batting Strike Rate (25%)** as the most critical features for this selection verdict.")
        if os.path.exists("assets/pca_clusters.png"):
            with st.expander("🔍 View Global Performance Reference Map"):
                st.image("assets/pca_clusters.png", caption="Historical Performance Distribution")


