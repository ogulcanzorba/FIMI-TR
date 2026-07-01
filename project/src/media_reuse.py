import numpy as np
import pandas as pd


def compute_media_reuse(df: pd.DataFrame, labels: np.ndarray) -> dict:
    results = {}
    clusters = [l for l in sorted(set(labels)) if l != -1]
    if "media_hash" not in df.columns:
        return {int(l): {"intra_overlap": 0, "cross_overlap": 0, "media_reuse_score": 0} for l in clusters}

    hash_to_clusters = {}
    for label in clusters:
        cdf = df[labels == label]
        for h in cdf["media_hash"].dropna().unique():
            if h:
                hash_to_clusters.setdefault(h, set()).add(label)

    for label in clusters:
        cdf = df[labels == label]
        hashes = cdf["media_hash"].dropna()
        hashes = hashes[hashes != ""]
        total = len(cdf)
        if total == 0:
            results[int(label)] = {"intra_overlap": 0, "cross_overlap": 0, "media_reuse_score": 0}
            continue

        hash_counts = hashes.value_counts()
        shared = hash_counts[hash_counts > 1].sum() if not hash_counts.empty else 0
        intra = shared / total

        cluster_hashes = set(hashes.unique())
        if len(cluster_hashes) > 0:
            cross_shared = sum(1 for h in cluster_hashes if len(hash_to_clusters.get(h, set())) > 1)
            cross = cross_shared / len(cluster_hashes)
        else:
            cross = 0

        score = 0.6 * intra + 0.4 * cross
        results[int(label)] = {
            "intra_overlap": round(intra, 4),
            "cross_overlap": round(cross, 4),
            "media_reuse_score": round(score, 4),
        }
    return results


def build_media_network(df: pd.DataFrame, labels: np.ndarray) -> dict:
    if "media_hash" not in df.columns:
        return {}
    clusters = [l for l in sorted(set(labels)) if l != -1]
    hash_to_clusters = {}
    for label in clusters:
        cdf = df[labels == label]
        for h in cdf["media_hash"].dropna().unique():
            if h:
                hash_to_clusters.setdefault(h, set()).add(label)

    edges = {}
    for h, cls in hash_to_clusters.items():
        cls_list = sorted(cls)
        for i in range(len(cls_list)):
            for j in range(i + 1, len(cls_list)):
                key = (cls_list[i], cls_list[j])
                edges[key] = edges.get(key, 0) + 1
    return edges
