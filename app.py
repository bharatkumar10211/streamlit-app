import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.encoder import encode_role,encode_format
import os

model_path = os.path.join("model", "selection_model.pkl")

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
            dismissals = st.number_input("Total Dismissals", min_value=0, value=0)

        avg_auto = runs / innings if innings > 0 else 0.0
        sr_auto = (runs / balls) * 100 if balls > 0 else 0.0
        
        # Automatically synced sliders with extremely high limits
        # Realistic slider limits as per Step 1
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
        
        # Safe access to variables that might be role-specific
        total_dismissals = dismissals if 'dismissals' in locals() else 0

        try:
            balls_val = balls if 'balls' in locals() else 0
            balls_per_innings = balls_val / innings if (innings > 0 and 'balls' in locals()) else 0
        except NameError:
            balls_per_innings = 0

        try:
            # Initialize defaults
            player_score = 0.0
            player_percentile = 0.0

            # ================= MINIMAL SANITY FILTER =================
            if innings < 5:
                label, color = "❌ NOT SELECTED (Insufficient Data)", "#94a3b8"
                # We still want to see potential scores for diagnostic reasons
                # but the label is already locked

            # ================= DYNAMIC DATASET-BASED SCORING =================
            df_full = pd.read_csv("final_player_scores.csv")
            df = df_full[(df_full["player_type"] == role) & (df_full["match_format"] == format_type)].copy()

            if df.empty:
                st.warning("⚠️ Category benchmark data is missing. Using fallback evaluation.")
                max_avg, max_sr, max_last5, max_hs = 50, 160, 200, 130
                max_wickets, max_econ, min_econ = 20, 12, 4
                max_bowl_avg, min_bowl_avg, max_dismiss = 45, 15, 25
                
                # Critical: Define percentile bands for fallback too
                if format_type == "T20": low_p, high_p = 0.80, 0.85
                elif format_type == "ODI": low_p, high_p = 0.67, 0.85
                else: low_p, high_p = 0.50, 0.85
                dataset_avg = 65.0
            else:
                # Use Quantiles to handle outliers and center the population
                max_avg = df['avg'].quantile(0.95)
                max_sr = df['sr'].quantile(0.95)
                max_last5 = df['last5_runs'].quantile(0.95)
                max_hs = df['high_score'].quantile(0.95)
                max_wickets = df['wickets'].quantile(0.95)
                max_econ = df['econ'].quantile(0.90)  # 90th percentile is Poor
                min_econ = df['econ'].quantile(0.10)  # 10th percentile is Elite
                max_bowl_avg = df['bowl_avg'].quantile(0.90)
                min_bowl_avg = df['bowl_avg'].quantile(0.10)
                # Ensure dismissals are safe (using catches column as base)
                df['total_d'] = df.get('catches', 0).fillna(0)
                max_dismiss = df['total_d'].quantile(0.95)

            def compute_dynamic_score(v_avg, v_sr, v_last5, v_hs, v_wickets, v_econ, v_bavg, v_dismiss, p_role, p_format):
                # Helper to clip normalized values between 0 and 1
                def norm(val, v_max):
                    return min(1.0, max(0.0, val / v_max)) if v_max > 0 else 0
                
                def inv_norm(val, v_max, v_min):
                    if v_max <= v_min: return 0.5
                    return min(1.0, max(0.0, (v_max - val) / (v_max - v_min)))

                if p_role == "Batsman":
                    n_avg, n_sr = norm(v_avg, max_avg), norm(v_sr, max_sr)
                    n_last5, n_hs = norm(v_last5, max_last5), norm(v_hs, max_hs)
                    if p_format == "T20":
                        return (n_avg * 0.3 + n_sr * 0.4 + n_last5 * 0.2 + n_hs * 0.1) * 100
                    elif p_format == "ODI":
                        return (n_avg * 0.5 + n_sr * 0.2 + n_last5 * 0.2 + n_hs * 0.1) * 100
                    else: # Test
                        return (n_avg * 0.7 + n_last5 * 0.2 + n_hs * 0.1) * 100

                elif p_role == "Bowler":
                    n_wickets = norm(v_wickets, max_wickets)
                    n_econ, n_bavg = inv_norm(v_econ, max_econ, min_econ), inv_norm(v_bavg, max_bowl_avg, min_bowl_avg)
                    if p_format == "T20":
                        return (n_wickets * 0.4 + n_econ * 0.4 + n_bavg * 0.2) * 100
                    elif p_format == "ODI":
                        return (n_wickets * 0.35 + n_econ * 0.35 + n_bavg * 0.3) * 100
                    else: # Test
                        return (n_wickets * 0.3 + n_bavg * 0.5 + n_econ * 0.2) * 100

                elif p_role == "All-Rounder":
                    n_avg, n_sr = norm(v_avg, max_avg), norm(v_sr, max_sr)
                    n_wickets, n_econ = norm(v_wickets, max_wickets), inv_norm(v_econ, max_econ, min_econ)
                    if p_format == "T20":
                        return (n_avg * 0.2 + n_sr * 0.3 + n_wickets * 0.3 + n_econ * 0.2) * 100
                    elif p_format == "ODI":
                        return (n_avg * 0.3 + n_sr * 0.2 + n_wickets * 0.3 + n_econ * 0.2) * 100
                    else: # Test
                        return (n_avg * 0.4 + n_wickets * 0.4 + n_econ * 0.1 + n_sr * 0.1) * 100

                else: # Wicketkeeper
                    n_avg, n_sr = norm(v_avg, max_avg), norm(v_sr, max_sr)
                    n_dismiss = norm(v_dismiss, max_dismiss)
                    if p_format == "T20":
                        return (n_avg * 0.3 + n_sr * 0.4 + n_dismiss * 0.3) * 100
                    elif p_format == "ODI":
                        return (n_avg * 0.4 + n_sr * 0.2 + n_dismiss * 0.4) * 100
                    else: # Test
                        return (n_avg * 0.5 + n_dismiss * 0.5) * 100

            # Calculate scores
            # Calculate scores
            player_score = compute_dynamic_score(avg, sr, last5, hs, wickets, econ, bowl_avg, total_dismissals, role, format_type)
            
            if not df.empty:
                df['score'] = df.apply(lambda row: compute_dynamic_score(
                    row['avg'], row['sr'], row['last5_runs'], row['high_score'],
                    row['wickets'], row['econ'], row['bowl_avg'], 
                    row.get('catches', 0), role, format_type
                ), axis=1)
                
                # Helper for Wicketkeeper charts
                if role == "Wicketkeeper":
                    df['total_d'] = df.get('catches', 0)

                # --- scouting Intelligence Logic ---
                if format_type == "T20":
                    low_p, high_p, elite_threshold = 0.65, 0.85, 88
                elif format_type == "ODI":
                    low_p, high_p, elite_threshold = 0.60, 0.85, 85
                else: # Test
                    low_p, high_p, elite_threshold = 0.50, 0.80, 75

                if not df.empty:
                    q_low = df['score'].quantile(low_p)
                    q_high = df['score'].quantile(high_p)
                    band_df = df[(df['score'] >= q_low) & (df['score'] <= q_high)]
                    dataset_avg = band_df['score'].mean() if not band_df.empty else df['score'].quantile((low_p + high_p)/2)
                    player_percentile = (df["score"] < player_score).sum() / len(df) * 100
                else:
                    dataset_avg, player_percentile = 60.0, 50.0

                # Final Classification
                if player_percentile >= elite_threshold:
                    label, color = "🌟 HIGHLY SELECTED", "#8b5cf6"
                elif player_percentile >= 70 and player_score >= dataset_avg:
                    label, color = "✅ SELECTED", "#22c55e"
                elif player_percentile >= 55:
                    label, color = "👍 RECOMMENDED", "#3b82f6"
                else:
                    label, color = "❌ NOT SELECTED", "#ef4444"

                # Define Role Metrics for Visuals & Reasoning
                if role == "All-Rounder":
                    x_col, y_col = "runs", "wickets"
                    x_label, y_label = "Total Runs", "Total Wickets"
                    p_x, p_y = runs, wickets
                    reverse_x, reverse_y = False, False
                elif role == "Batsman":
                    x_col, y_col = "sr", "avg"
                    x_label, y_label = "Strike Rate", "Batting Average"
                    p_x, p_y = sr, avg
                    reverse_x, reverse_y = False, False
                elif role == "Bowler":
                    x_col, y_col = "econ", "wickets"
                    x_label, y_label = "Economy Rate (Lower is Better)", "Total Wickets"
                    p_x, p_y = econ, wickets
                    reverse_x, reverse_y = True, False
                else: # Wicketkeeper
                    x_col, y_col = "total_d", "avg"
                    x_label, y_label = "Total Dismissals", "Batting Average"
                    p_x, p_y = total_dismissals, avg
                    reverse_x, reverse_y = False, False

                st.write("---")
                st.write(f"**Diagnostic View** -> Player Score: `{player_score:.2f}` | Selection Baseline ({int(low_p*100)}-{int(high_p*100)}%): `{dataset_avg:.2f}`")

        except Exception as e:
            st.error(f"Error in selection pipeline: {e}")
            label, color, player_percentile = "ERROR", "#991b1b", 0.0

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
                # Role-Aware Radar Chart
                if role == "Batsman":
                    labels = ["Avg", "SR", "Form", "Peak"]
                    user_prof = [min(avg, max_avg)/max_avg*100, min(sr, max_sr)/max_sr*100, 
                                 min(last5, max_last5)/max_last5*100, min(hs, max_hs)/max_hs*100]
                elif role == "Bowler":
                    labels = ["Wkts", "Econ", "BAvg", "Form"]
                    user_prof = [min(wickets, max_wickets)/max_wickets*100, (1 - min(econ, max_econ)/max_econ)*100,
                                 (1 - min(bowl_avg, max_bowl_avg)/max_bowl_avg)*100, min(last5, 100)/100*100]
                elif role == "All-Rounder":
                    labels = ["Bt Avg", "SR", "Wkts", "Econ"]
                    user_prof = [min(avg, max_avg)/max_avg*100, min(sr, max_sr)/max_sr*100,
                                 min(wickets, max_wickets)/max_wickets*100, (1 - min(econ, max_econ)/max_econ)*100]
                else: # Wicketkeeper
                    labels = ["Avg", "SR", "Dism", "Form"]
                    user_prof = [min(avg, max_avg)/max_avg*100, min(sr, max_sr)/max_sr*100,
                                 min(total_dismissals, max_dismiss)/max_dismiss*100, min(last5, 100)/100*100]

                bench_prof = [60, 60, 60, 60] # Professional baseline
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
                ax_radar.set_xticklabels(labels, color="#94a3b8", fontsize=9)
                ax_radar.set_yticklabels([]) 

                ax_radar.tick_params(colors='white')

                ax_radar.set_title("Player Profile", color="white", fontsize=10)

                st.pyplot(fig_radar)

            with gc2:
                # Role-Specific Performance Matrix
                st.markdown(f"#### {role} Performance Matrix")
                if not df.empty and 'x_col' in locals():
                    # Create Plotly Scatter using pre-defined metrics
                    fig_scatter = px.scatter(
                        df, x=x_col, y=y_col,
                        hover_name="player_name",
                        color_discrete_sequence=["#475569"],
                        opacity=0.4,
                        template="plotly_dark",
                        labels={x_col: x_label, y_col: y_label}
                    )
                    
                    # Force Elite Quadrant (Top-Right)
                    if reverse_x: fig_scatter.update_xaxes(autorange="reversed")
                    if reverse_y: fig_scatter.update_yaxes(autorange="reversed")
                    
                    # Add current player
                    fig_scatter.add_scatter(
                        x=[p_x], y=[p_y],
                        mode="markers+text",
                        marker=dict(size=18, color=color, symbol="star", line=dict(width=2, color="white")),
                        text=["YOU"], textposition="top center",
                        name="Target Player"
                    )
                    
                    # Add Average Lines
                    fig_scatter.add_vline(x=df[x_col].mean(), line_dash="dash", line_color="#94a3b8")
                    fig_scatter.add_hline(y=df[y_col].mean(), line_dash="dash", line_color="#94a3b8")
                    
                    fig_scatter.update_layout(
                        showlegend=False, height=400,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info("Performance matrix unavailable (Missing Benchmarks)")

            # --- 3. Final Selection Intelligence ---
            st.write("---")
            st.markdown(f"### 📍 Selection Reasons")
            
            if not df.empty:
                reasons = []
                # Use the dynamic elite_threshold defined in the scouting logic
                if 'elite_threshold' in locals() and player_percentile >= elite_threshold:
                    reasons.append(f"🏆 **Elite Potential**: Ranks in the top {100-int(player_percentile)}% of the professional {role} population.")
                elif player_percentile >= 70:
                    reasons.append(f"✅ **Strong Performer**: Ranks above 70% of the population, showing consistent professional standards.")
                
                if 'p_x' in locals() and 'x_col' in locals() and p_x > df[x_col].mean():
                    reasons.append(f"📈 **Superior {x_label}**: Outperforming the national baseline for your primary skill.")
                
                if 'p_y' in locals() and 'y_col' in locals():
                    if y_col == "econ" and p_y < df[y_col].mean():
                        reasons.append(f"📉 **Control Discipline**: Exceptional economy rate compared to other {role}s.")
                    elif y_col != "econ" and p_y > df[y_col].mean():
                        reasons.append(f"🔥 **High Impact {y_label}**: Higher volume contribution than the population benchmark.")
                
                if not reasons:
                    reasons.append("⚠️ **Development Needed**: Metrics currently sit below the required professional averages for selection.")
                    
                for r in reasons: st.write(r)
            else:
                st.info("Selection reasons unavailable (Missing Benchmarks)")

        st.divider()
        st.info("🚀 **Dataset-Driven Selection Engine**: This verdict is calculated by comparing your **Final AI Score** against the historical performance of all players in this category using **Percentile Ranks**.")
        if os.path.exists("assets/pca_clusters.png"):
            with st.expander("🔍 View Global Performance Reference Map"):
                st.image("assets/pca_clusters.png", caption="Historical Performance Distribution")


