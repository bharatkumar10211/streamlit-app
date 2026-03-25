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
        matches = len(group)
        
        return pd.Series({
            'avg': total_runs / (innings if innings > 0 else 1),
            'sr': (total_runs / (total_balls if total_balls > 0 else 1)) * 100,
            'last5_runs': last5,
            'high_score': high_score,
            'wickets': wickets,
            'econ': econ,
            'bowl_avg': bowl_avg,
            'matches': matches,
            'runs': total_runs
        })

    print("Engineering performance-oriented features...")
    agg_df = raw_df.groupby(['player_name', 'country', 'player_type', 'match_format']).apply(get_player_enhanced_stats).reset_index()

    # 3. Refined Scoring Formula (Batsman centric as per Step 5)
    # score = avg*0.35 + sr*0.25 (scaled) + last5*0.3 + hs*0.1
    # We normalize internal components for a balanced score
    def calculate_balanced_score(row):
        # Normalizing Strike Rate (ideal 140+ in T20)
        norm_sr = min(row['sr'], 200) / 2
        # Normalizing Last 5 (ideal 200+)
        norm_last5 = min(row['last5_runs'], 250) / 2.5
        # Normalizing High Score (ideal 100+)
        norm_hs = min(row['high_score'], 120) / 1.2
        # Average (ideal 45+)
        norm_avg = min(row['avg'], 60) * 1.66
        
        if row['player_type'] in ['Batsman', 'Wicketkeeper']:
            s = (norm_avg * 0.35) + (norm_sr * 0.25) + (norm_last5 * 0.30) + (norm_hs * 0.10)
        elif row['player_type'] == 'Bowler':
            # Bowlers need different metric normalization
            norm_wickets = min(row['wickets'], 50) * 2
            norm_econ = max(15 - row['econ'], 0) * 6.6
            s = (norm_wickets * 0.4) + (norm_econ * 0.4) + (norm_last5 * 0.2)
        else: # All-Rounder
            s = (norm_avg * 0.2) + (norm_sr * 0.15) + (norm_last5 * 0.2) + (min(row['wickets'], 20) * 2.5)
            
        return s

    agg_df['score'] = agg_df.apply(calculate_balanced_score, axis=1)

    # 4. Improved Clustering (StandardScaler + Performance Features)
    # Using quality features ONLY
    BATS_FEATURES = ['avg', 'sr', 'last5_runs', 'high_score']
    # For global clustering we use a mix but keep them quality focused
    X_clustering = agg_df[['avg', 'sr', 'last5_runs', 'high_score', 'wickets', 'econ']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clustering)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    agg_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to levels based on mean score
    c_means = agg_df.groupby('cluster')['score'].mean().sort_values().index
    agg_df['levels'] = agg_df['cluster'].map({c_means[0]: 0, c_means[1]: 1, c_means[2]: 2})

    # 5. Top 16 Benchmarking per Country
    benchmarks = agg_df.sort_values("score", ascending=False) \
                    .groupby(["country", "match_format", "player_type"]) \
                    .head(16) \
                    .groupby(["country", "match_format", "player_type"])["score"] \
                    .mean().to_dict()

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
    
    # Save the Random Forest classifier
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_scaled, agg_df['levels'])
    joblib.dump(rf, "model/selection_model.pkl")

    print("Success: Refined pipeline with High Score and Form impact complete.")

if __name__ == "__main__":
    build_refined_pipeline()