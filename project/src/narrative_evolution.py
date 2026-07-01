import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from src.nlp_pipeline import compute_embeddings, load_model


def compute_evolution(
    df: pd.DataFrame,
    cluster_id: int,
    labels: np.ndarray,
    all_embeddings: np.ndarray,
    window_hours: float = 2.0,
) -> list[dict]:
    cluster_mask = labels == cluster_id
    cdf = df[cluster_mask].copy()
    if len(cdf) < 3:
        return []

    cdf["timestamp"] = pd.to_datetime(cdf["timestamp"])
    cdf = cdf.sort_values("timestamp")

    t_min = cdf["timestamp"].min()
    t_max = cdf["timestamp"].max()
    total_hours = (t_max - t_min).total_seconds() / 3600
    if total_hours < window_hours:
        window_hours = max(total_hours / 3, 0.5)

    n_windows = max(int(total_hours / window_hours), 2)
    windows = [t_min + pd.Timedelta(hours=i * window_hours) for i in range(n_windows + 1)]

    global_idx = np.where(cluster_mask)[0]
    local_to_global = dict(enumerate(global_idx))

    evolution_steps = []
    for i in range(len(windows) - 1):
        mask = (cdf["timestamp"] >= windows[i]) & (cdf["timestamp"] < windows[i + 1])
        wdf = cdf[mask]
        if len(wdf) == 0:
            continue

        local_positions = np.where(mask.values)[0]
        w_global_positions = [local_to_global[p] for p in local_positions]
        w_embs = all_embeddings[w_global_positions]

        if len(w_embs) == 0:
            continue

        centroid = w_embs.mean(axis=0).reshape(1, -1)
        dists = cosine_distances(centroid, w_embs).flatten()
        best_idx = dists.argmin()
        representative_text = wdf.iloc[best_idx]["text"]

        if i > 0:
            prev_mask = (cdf["timestamp"] >= windows[i - 1]) & (cdf["timestamp"] < windows[i])
            prev_local = np.where(prev_mask.values)[0]
            prev_global = [local_to_global[p] for p in prev_local]
            prev_embs = all_embeddings[prev_global]
            if len(prev_embs) > 0:
                prev_centroid = prev_embs.mean(axis=0).reshape(1, -1)
                change = float(cosine_distances(centroid, prev_centroid).flatten()[0])
            else:
                change = 0.0
        else:
            change = 0.0

        window_start = windows[i].strftime("%H:%M") if total_hours < 48 else windows[i].strftime("%d.%m %H:%M")
        time_display = f"{window_start}~{windows[i+1].strftime('%H:%M')}"

        evolution_steps.append({
            "time": time_display,
            "text": representative_text,
            "change_score": round(change, 4),
            "post_count": len(wdf),
        })

    return evolution_steps


def compute_all_evolutions(
    df: pd.DataFrame,
    labels: np.ndarray,
    min_posts: int = 5,
) -> dict[int, list[dict]]:
    model = load_model()
    all_embeddings = compute_embeddings(df["text"].tolist(), model)

    results = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        n = int((labels == label).sum())
        if n < min_posts:
            continue
        evo = compute_evolution(df, label, labels, all_embeddings)
        if evo:
            results[int(label)] = evo
    return results
