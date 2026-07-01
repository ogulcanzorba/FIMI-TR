import sys, os
sys.path.insert(0, "project")
import pandas as pd
import numpy as np
from src.coordination import compute_all_signals, build_interaction_network

df = pd.read_csv("project/data/synthetic_data.csv", parse_dates=["timestamp", "account_created_at"])
from src.nlp_pipeline import run_pipeline
labels, tsne, summaries = run_pipeline(df, eps=0.45, min_samples=2)
labels = np.array(labels)

sig = compute_all_signals(df, labels)
print(f"Signals computed for {len(sig)} clusters")
for cid, s in sorted(sig.items())[:3]:
    print(f"  Kume {cid}: sync={s['synchrony']:.2f} age={s['age_clustering']:.2f} txt={s['text_account_ratio']:.2f} div={s['account_diversity']:.2f} coord={s['coordination_score']:.2f}")

G = build_interaction_network(df, labels, min_edge_weight=1)
print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print("OK")
