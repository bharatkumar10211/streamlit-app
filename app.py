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

plt.style.use("dark_background")

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
        background: #0f172a;
        color: #e2e8f0;   /* changed */
    }

    /* Premium Result Card */
    .result-card {
        background: rgba(255, 255, 255, 0.9);
        color: #0f172a;   /* changed for contrast */
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Sleek Action Button */
    .stButton>button {
        background: #3b82f6;   /* changed */
        color: white;
    }
    .stButton>button:hover {
        background: #2563eb;   /* changed */
    }

    /* Headers */
    h1, h2, h3 {
        color: #93c5fd !important;   /* changed */
    }

    /* Labels (important fix) */
    label {
        color: #cbd5f5 !important;   /* changed */
    }

</style>
""", unsafe_allow_html=True)

st.title("🏏 Premium Cricket Squad AI")
status_placeholder = st.empty()
st.write("---")
# Benchmarks removed in favor of percentile-based selection
# max_score removed as per Step 1 instruction

# Top Level Selectors (Mobile Responsive)
c1, c2 = st.columns(2)
with c1:
    role = st.selectbox("Player Role", ["Batsman", "Bowler", "Wicketkeeper", "All-Rounder"])
with c2:
    format_type = st.selectbox("Format", ["T20", "ODI", "Test"])

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
        # Realistic slider limits as per Step 1
        st.write("---")
        final_avg = st.slider("Average (Auto-Calculated Slider)", 0.0, 60.0, value=float(min(avg_auto, 60.0)))
        final_sr = st.slider("Strike Rate (Auto-Calculated Slider)", 0.0, 200.0, value=float(min(sr_auto, 200.0)))
        
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
        # Realistic bowling limits
        final_econ = st.slider("Economy Rate", 0.0, 12.0, value=float(min(econ_auto, 12.0)))
        final_avg_bowl = st.slider("Bowling Average", 0.0, 60.0, value=float(min(bowl_avg_auto, 60.0)))
        
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
        
        # ================= FEATURE PREPARATION =================
        avg, sr, last5, hs = batting_stats['avg'], batting_stats['sr'], batting_stats['last5'], batting_stats['hs']
        wickets, econ = bowling_stats['wickets'], bowling_stats['econ']
        bowl_avg = bowling_stats['bowl_avg']
        
        # Safe access to variabels that might be role-specific
        try:
            total_dismissals = (dismissals if 'dismissals' in locals() else 0) + (stumpings if 'stumpings' in locals() else 0)
        except NameError:
            total_dismissals = 0

        try:
            balls_val = balls if 'balls' in locals() else 0
            balls_per_innings = balls_val / innings if (innings > 0 and 'balls' in locals()) else 0
        except NameError:
            balls_per_innings = 0

        # ================= HYBRID SCORING =================
        # 1. Prepare input (batting base)
        input_features = np.array([[avg, sr, last5, hs]])

        try:
            scaler = joblib.load("model/scaler.pkl")
            input_scaled = scaler.transform(input_features)

            proba = model.predict_proba(input_scaled)[0]

            # Better weighted score (Elite=95, Avg=75, Low=50)
            model_score = (proba[0] * 50) + (proba[1] * 75) + (proba[2] * 95)

            # 2. Benchmark usage removed as per Step 2.1

            # ================= HARD FILTERS (Step 2 & 7) =================
            reject_flag = False
            if role == "Batsman":
                if avg < 20 or sr < 100:
                    reject_flag = True
            elif role == "Bowler":
                if econ > 10 or bowl_avg > 45:
                    reject_flag = True
            elif role == "All-Rounder":
                if avg < 20 and wickets < 5:
                    reject_flag = True
            elif role == "Wicketkeeper":
                if avg < 20 and total_dismissals < 10:
                    reject_flag = True

            # ================= ROLE BOOST (PENALTY BASED) =================
            role_boost = 0
            if role == "Bowler":
                wicket_score = wickets * 1.5
                econ_score = (10 - econ) * 3          # penalty included
                avg_score = (50 - bowl_avg) * 1.5     # penalty included
                role_boost = (wicket_score + econ_score + avg_score) * 0.2

            elif role == "All-Rounder":
                bat_part = (avg - 30) * 1.5 + (sr - 120) * 1.0
                bowl_part = (wickets * 1.5) + ((10 - econ) * 2)
                balance_penalty = abs(bat_part - bowl_part) * 0.3
                role_boost = (bat_part + bowl_part - balance_penalty) * 0.15

            elif role == "Wicketkeeper":
                bat_score = (avg - 30) * 1.5 + (sr - 120) * 1.0
                keeping_score = (total_dismissals - 20) * 1.2
                role_boost = (bat_score + keeping_score) * 0.1

            elif role == "Batsman":
                avg_score = (avg - 30) * 2            # below 30 → penalty
                sr_score = (sr - 120) * 1.5           # below 120 → penalty
                form_score = (last5 - 150) * 0.5      # poor form → penalty
                hs_score = (hs - 50) * 0.3            # low impact → penalty
                role_boost = (avg_score + sr_score + form_score + hs_score) * 0.1

            # Safety cap for role_boost
            role_boost = max(min(role_boost, 30), -30)

            # ================= HYBRID FINAL (Penalty Enabled) =================
            player_score = (0.7 * model_score) + (0.3 * role_boost)
            player_score = max(min(player_score, 100), 0) # Normalization

            # ================= DATASET COMPARISON (Fixed to use Aggr Dataset) =================
            df_full = pd.read_csv("final_player_scores.csv")
            df = df_full[(df_full["player_type"] == role) & (df_full["match_format"] == format_type)].copy()

            def compute_score_internal(row_data, player_role):
                avg_val = row_data.get("avg", 0)
                sr_val = row_data.get("sr", 0)
                wickets_val = row_data.get("wickets", 0)
                econ_val = row_data.get("econ", 0)
                bowl_avg_val = row_data.get("bowl_avg", 0)
                last5_val = row_data.get("last5_runs", 0)
                hs_val = row_data.get("high_score", 0)

                if player_role == "Batsman":
                    return (avg_val - 30) * 2 + (sr_val - 120) * 1.5 + (last5_val - 150) * 0.5 + (hs_val - 50) * 0.3
                elif player_role == "Bowler":
                    return (wickets_val * 1.5) + (10 - econ_val) * 3 + (50 - bowl_avg_val) * 1.5
                elif player_role == "All-Rounder":
                    bat = (avg_val - 30) * 1.5 + (sr_val - 120) * 1.0
                    bowl = (wickets_val * 1.5) + (10 - econ_val) * 2
                    return (bat + bowl) * 0.5
                else: # Wicketkeeper
                    dismissals_val = row_data.get("catches", 0) + row_data.get("stumpings", 0)
                    return (avg_val - 30) * 1.5 + (sr_val - 120) * 1.0 + (dismissals_val - 20) * 1.2

            # Compute dataset scores
            df["stat_score"] = df.apply(lambda r: compute_score_internal(r, role), axis=1)
            
            # Scale dataset scores to be comparable with the hybrid player_score (0.7 * 75 average model score + 0.3 * stat)
            df["hybrid_score_calc"] = (0.7 * 75) + (0.3 * df["stat_score"])
            
            # Compute percentile rank (MAIN LOGIC)
            if not df.empty:
                # Compare hybrid vs hybrid for fair ranking
                player_percentile = (df["hybrid_score_calc"] < player_score).mean() * 100
            else:
                player_percentile = 50.0 

        except Exception as e:
            st.error(f"Error in calculation: {e}")
            player_percentile = 0.0

        # ================= FINAL LABEL =================
        # ================= FINAL LABEL (Clean & Filtered) =================
        if reject_flag:
            label, color = "❌ NOT SELECTED", "#ef4444"
        elif player_percentile >= 80:
            label, color = "🌟 SELECTED", "#22c55e"
        elif player_percentile >= 65:
            label, color = "👍 RECOMMENDED", "#3b82f6"
        elif player_percentile >= 45:
            label, color = "⚠️ AVERAGE", "#facc15"
        else:
            label, color = "❌ NOT SELECTED", "#ef4444"

        # Unified Selection Dashboard Panel
        st.write("---")
        with st.container():
            st.markdown(f"""
<div style='background: #1e293b; padding: 30px; border-radius: 20px; border: 1px solid #334155; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center; margin-bottom: 25px;'>
    <h4 style='color: #cbd5f5; margin-bottom: 10px;'>Selection Verdict</h4>
    <h1 style='color: {color}; font-size: 3.2rem; margin: 0;'>{label}</h1>
    <p style='color: #94a3b8; font-size: 1.1rem;'>
        Percentile Rank: <b style='color:white;'>{player_percentile:.1f}%</b>
    </p>
    <div style='display: flex; justify-content: space-around; background: #0f172a; padding: 15px; border-radius: 12px; margin: 20px 0;'>
        <div style='text-align: center;'>
            <span style='color: #94a3b8; font-size: 0.9rem;'>RECENT FORM</span><br>
            <b style='font-size: 1.5rem; color:white;'>{last5}</b>
        </div>
        <div style='text-align: center;'>
            <span style='color: #94a3b8; font-size: 0.9rem;'>PEAK IMPACT</span><br>
            <b style='font-size: 1.5rem; color:white;'>{hs}</b>
        </div>
        <div style='text-align: center;'>
            <span style='color: #94a3b8; font-size: 0.9rem;'>CONFIDENCE</span><br>
            <b style='color: {color}; font-size: 1.5rem;'>
                {player_percentile:.1f}%
            </b>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

            # --- 1. Selection Probability (Top Header) ---
            prob_val = player_percentile
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_val,
                title={'text': "Percentile Rank (%)", 'font': {'size': 18}},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color}, 'steps': [
                    {'range': [0, 40], 'color': '#ef4444'},
                    {'range': [40, 60], 'color': '#facc15'},
                    {'range': [60, 75], 'color': '#3b82f6'},
                    {'range': [75, 100], 'color': '#22c55e'}
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

                # Better styling
                ax_radar.set_facecolor("#0f172a")
                fig_radar.patch.set_facecolor("#0f172a")

                # Plot
                ax_radar.plot(angles, user_prof, linewidth=2, color=color)
                ax_radar.fill(angles, user_prof, alpha=0.3, color=color)

                ax_radar.plot(angles, bench_prof, linestyle='dashed', linewidth=1, color='#bdc3c7')

                # Labels
                ax_radar.set_xticks(angles[:-1])
                ax_radar.set_xticklabels(labels, fontsize=9, color="white")

                ax_radar.tick_params(colors='white')

                ax_radar.set_title("🏆 Player Profile", color="white", fontsize=10)

                st.pyplot(fig_radar)

            with gc2:
                # 3. Enhanced Dynamic PCA highlighting with Zone Intelligence
                def classify_zone(x, y):
                    if x > 60 and y > 60: return "ELITE", "success"
                    elif x > 48 and y > 48: return "ABOVE AVG", "info"
                    else: return "BELOW AVG", "error"

                user_x, user_y = player_percentile, (last5/2.5 + hs)/2.0
                z_lab, z_typ = classify_zone(user_x, user_y)
                st.markdown(f"<div style='text-align: center; background: #f8f9fa; padding: 5px; border-radius: 5px; border-left: 5px solid {color}; margin-bottom: 15px;'><b>{z_lab} ZONE</b></div>", unsafe_allow_html=True)

                try:
                    df = pd.read_csv(f"player_stats_{format_type.lower()}.csv")
                    df = df[df["player_type"] == role]

                    if role in ["Batsman", "Wicketkeeper"]:
                        x = df["runs"]
                        y = df["batting_strike_rate"]
                        xlabel = "Runs"
                        ylabel = "Strike Rate"
                        highlight_x = (runs if 'runs' in locals() else 0)
                        highlight_y = sr
                    elif role == "Bowler":
                        x = df["wickets"]
                        y = df["economy"]
                        xlabel = "Wickets"
                        ylabel = "Economy"
                        highlight_x = wickets
                        highlight_y = econ
                    else: # All-Rounder
                        x = df["runs"]
                        y = df["wickets"]
                        xlabel = "Runs"
                        ylabel = "Wickets"
                        highlight_x = (runs if 'runs' in locals() else 0)
                        highlight_y = wickets
                    
                    fig_cluster, ax_cluster = plt.subplots(figsize=(4, 4))
                    
                    # Better styling
                    ax_cluster.set_facecolor("#0f172a")
                    fig_cluster.patch.set_facecolor("#0f172a")
                    
                    ax_cluster.scatter(x, y, alpha=0.4, color='#3b82f6')
                    
                    # Highlight player
                    ax_cluster.scatter([highlight_x], [highlight_y], s=150, color=color, edgecolors='white', linewidth=2, zorder=5)
                    
                    ax_cluster.set_title(f"{role} Performance Distribution", color="white", fontsize=10)
                    ax_cluster.set_xlabel(xlabel, color="white")
                    ax_cluster.set_ylabel(ylabel, color="white")
                    ax_cluster.tick_params(colors="white")
                    
                    st.pyplot(fig_cluster)
                    
                    # Meaning interpretation
                    if reject_flag:
                        st.error("Player rejected due to poor core performance metrics")
                    elif player_percentile >= 80:
                        st.success("Player is in the Elite cluster")
                    elif player_percentile >= 65:
                        st.info("Player is in the Recommended cluster")
                    else:
                        st.warning("Player is in the Average or below cluster")
                except Exception as e:
                    st.info("Intelligence clustering analysis complete.")

            # --- 4. Explanation System ---
            st.write("---")
            st.subheader("📌 Selection Reasons")
            reasons = []
            if not reject_flag:
                if avg > 40: reasons.append("Strong batting average")
                if sr > 130: reasons.append("High strike rate")
                if role == "Bowler" and econ < 7: reasons.append("Excellent economy rate")
                if role == "Bowler" and wickets > 15: reasons.append("Good wicket-taking ability")
                if player_percentile > 75: reasons.append("Top percentile performer")
                if role == "All-Rounder" and abs((avg-30)*1.5 - (wickets*1.5)) < 10: reasons.append("Highly balanced all-round capability")
                
                if not reasons: reasons.append("Consistent performance across primary metrics")
            else:
                if role == "Batsman" and avg < 20: reasons.append("Extremely low batting average")
                if role == "Batsman" and sr < 100: reasons.append("Below par strike rate")
                if role == "Bowler" and econ > 10: reasons.append("High economy rate")
                if role == "Bowler" and bowl_avg > 45: reasons.append("Poor bowling average")
                if role == "All-Rounder" and avg < 20: reasons.append("Weak batting contribution")
                if role == "Wicketkeeper" and total_dismissals < 10: reasons.append("Low stumping/catch impact")

            for r in reasons:
                st.write(f"- {r}")

        st.divider()
        st.info("🚀 **Dataset-Driven Selection Engine**: This verdict is calculated by comparing your **Final AI Score** against the historical performance of all players in this category using **Percentile Ranks**.")
        if os.path.exists("assets/pca_clusters.png"):
            with st.expander("🔍 View Global Performance Reference Map"):
                st.image("assets/pca_clusters.png", caption="Historical Performance Distribution")


