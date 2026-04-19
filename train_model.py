import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import matplotlib.pyplot as plt

def build_refined_pipeline():
    # 1. Load and Combine
    files = {"T20": "player_stats_t20.csv", "ODI": "player_stats_odi.csv", "Test": "player_stats_test.csv"}
    all_raw_dfs = []
    for fmt, file in files.items():
        if os.path.exists(file):
            rdf = pd.read_csv(file)
            rdf['match_format'] = fmt
            all_raw_dfs.append(rdf)
    
    if not all_raw_dfs: return
    raw_df = pd.concat(all_raw_dfs, ignore_index=True).fillna(0)
    raw_df['match_date'] = pd.to_datetime(raw_df['match_date'], errors='coerce')

    # 2. Advanced Player Feature Engineering
    # Calculate Last 5 runs and High Score per player context
    def get_player_enhanced_stats(group):
        sorted_group = group.sort_values('match_date', ascending=False)
        last5 = sorted_group.head(5)['runs'].sum()
        high_score = group['runs'].max()
        innings = group['batted'].sum()
        total_runs = group['runs'].sum()
        total_balls = group['balls'].sum()
        wickets = group['wickets'].sum()
        econ = group['economy'].mean()
        bowl_avg = group['bowling_average'].mean()
        catches = group['catches'].sum() if 'catches' in group else 0
        stumpings = group['stumpings'].sum() if 'stumpings' in group else 0
        matches = len(group)
        
        return pd.Series({
            'avg': total_runs / (innings if innings > 0 else 1),
            'sr': (total_runs / (total_balls if total_balls > 0 else 1)) * 100,
            'last5_runs': last5,
            'high_score': high_score,
            'wickets': wickets,
            'econ': econ,
            'bowl_avg': bowl_avg,
            'catches': catches,
            'stumpings': stumpings,
            'matches': matches,
            'runs': total_runs,
            'total_balls_faced': total_balls,
            'balls_per_innings': total_balls / (innings if innings > 0 else 1)
        })

    print("Engineering performance-oriented features...")
    agg_df = raw_df.groupby(['player_name', 'player_type', 'match_format']).apply(get_player_enhanced_stats).reset_index()

    # 3. Refined Scoring Formula (Batsman centric as per Step 5)
    # score = avg*0.35 + sr*0.25 (scaled) + last5*0.3 + hs*0.1
    # We normalize internal components for a balanced score
    def get_base_score(row):
        avg, sr, wickets, econ = row['avg'], row['sr'], row['wickets'], row['econ']
        bowl_avg, last5, hs = row['bowl_avg'], row['last5_runs'], row['high_score']
        dismissals = row.get('catches', 0) + row.get('stumpings', 0)
        role = row['player_type']

        if role == "Batsman":
            s = (avg / 60) * 40 + (sr / 180) * 30 + (last5 / 250) * 20 + (hs / 150) * 10
        elif role == "Bowler":
            s = (wickets / 20) * 40 + ((10 - econ) / 10) * 30 + ((50 - bowl_avg) / 50) * 30
        elif role == "All-Rounder":
            bat = (avg / 60) * 50 + (sr / 180) * 50
            bowl = (wickets / 20) * 50 + ((10 - econ) / 10) * 50
            s = (bat * 0.5) + (bowl * 0.5)
        else: # Wicketkeeper
            s = (avg / 60) * 40 + (sr / 180) * 40 + (dismissals / 30) * 20
        
        return s # Raw score with penalties

    def calibrate_score(row):
        # Calibration removed - using raw scores for training
        return row['score']

    def apply_max_cap(value, role, fmt):
        return value # No caps for raw training scores

    print("Calculating base scores and applying calibration & caps...")
    agg_df['score'] = agg_df.apply(get_base_score, axis=1)
    agg_df['score'] = agg_df.apply(calibrate_score, axis=1)
    agg_df['score'] = agg_df.apply(lambda x: apply_max_cap(x['score'], x['player_type'], x['match_format']), axis=1)

    # 4. Improved Clustering (StandardScaler + Performance Features)
    # Using quality features only to avoid mixing batsman and bowler signals
    BATS_FEATURES = ['avg', 'sr', 'last5_runs', 'high_score']
    X_clustering = agg_df[['avg', 'sr', 'last5_runs', 'high_score']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clustering)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    agg_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to levels based on mean score
    c_means = agg_df.groupby('cluster')['score'].mean().sort_values().index
    agg_df['levels'] = agg_df['cluster'].map({c_means[0]: 0, c_means[1]: 1, c_means[2]: 2})

    # 5. Top 16 Benchmarking Overall (Percent)
    agg_df["percent"] = agg_df["score"] 
    max_score = agg_df["score"].max() # Still needed for some legacy compatibility maybe, but percent is now absolute

    # Define base benchmarks based on realistic cricket selection standards
    BASE_BENCHMARKS = {
        ("T20", "Batsman"): 65,
        ("T20", "Wicketkeeper"): 64,
        ("T20", "Bowler"): 62,
        ("T20", "All-Rounder"): 65,

        ("ODI", "Batsman"): 65,
        ("ODI", "Wicketkeeper"): 63,
        ("ODI", "Bowler"): 64,
        ("ODI", "All-Rounder"): 65,

        ("Test", "Batsman"): 60,
        ("Test", "Wicketkeeper"): 60,
        ("Test", "Bowler"): 60,
        ("Test", "All-Rounder"): 60,
    }

    def apply_role_caps(value, role, fmt):
        return min(value, 100)

    benchmarks = {}
    roles = agg_df["player_type"].unique()
    formats = agg_df["match_format"].unique()

    for f in formats:
        for r in roles:
            filtered = agg_df[
                (agg_df["player_type"] == r) &
                (agg_df["match_format"] == f)
            ]
            
            if len(filtered) > 0:
                benchmark = np.percentile(filtered["percent"], 70)
                # Apply hard role caps to prevent excessively high benchmarks
                benchmark = apply_role_caps(benchmark, r, f)
            else:
                benchmark = 0
                
            benchmarks[(f, r)] = benchmark

    # Add percentile score (IMPORTANT) - Step 1.3
    agg_df["percentile_score"] = agg_df.groupby(
        ["player_type", "match_format"]
    )["score"].rank(pct=True) * 100

    # Save dataset with score - Step 1.4
    agg_df.to_csv("final_player_scores.csv", index=False)

    # 6. Advanced Visualization Suite
    os.makedirs("assets", exist_ok=True)
    
    # Improved PCA Graph
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10,7))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=agg_df['levels'], cmap='viridis', alpha=0.6)
    plt.title("Player Clusters (Improved PCA Performance Axis)")
    plt.xlabel("Quality & Consistency Axis")
    plt.ylabel("Output Volatility Axis")
    plt.colorbar(label="Performance Tier (0=Low, 2=High)")
    plt.savefig("assets/pca_clusters.png")
    
    # Improved Histogram
    plt.figure(figsize=(10,5))
    plt.hist(agg_df['score'], bins=30, color='royalblue', edgecolor='white', alpha=0.8)
    plt.title("Global Player Score Distribution")
    plt.xlabel("Model score (0-100)")
    plt.ylabel("Number of Players")
    plt.savefig("assets/score_dist.png")

    # NEW: Form vs Performance Graph
    plt.figure(figsize=(10,6))
    plt.scatter(agg_df['last5_runs'], agg_df['score'], c=agg_df['levels'], cmap='magma', alpha=0.4)
    plt.title("Recent Form Impact on Final Score")
    plt.xlabel("Recent Form (Last 5 Innings Runs)")
    plt.ylabel("Confidence Score")
    plt.savefig("assets/form_vs_score.png")

    # 7. Save Models
    os.makedirs("model", exist_ok=True)
    joblib.dump(kmeans, "model/kmeans_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(benchmarks, "model/benchmarks.pkl")
    joblib.dump(max_score, "model/max_score.pkl")
    
    # Save the Random Forest classifier
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, agg_df['levels'])
    joblib.dump(rf, "model/selection_model.pkl")

    print("Success: Refined pipeline with High Score and Form impact complete.")

if __name__ == "__main__":
    build_refined_pipeline()