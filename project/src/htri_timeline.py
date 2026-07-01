import numpy as np
import pandas as pd
from src.coordination import compute_synchrony, compute_account_age_clustering, compute_text_account_ratio, compute_account_diversity, compute_acceleration
from src.media_reuse import compute_media_reuse


def compute_cluster_htri_timeline(df: pd.DataFrame, labels: np.ndarray, n_windows: int = 8) -> dict:
    media_reuse_cache = {}
    hash_to_clusters = {}
    clusters = [l for l in sorted(set(labels)) if l != -1]
    if "media_hash" in df.columns:
        for label in clusters:
            cdf = df[labels == label]
            for h in cdf["media_hash"].dropna().unique():
                if h:
                    hash_to_clusters.setdefault(h, set()).add(label)

    results = {}
    for label in clusters:
        cdf = df[labels == label].copy()
        cdf["timestamp"] = pd.to_datetime(cdf["timestamp"])
        cdf = cdf.sort_values("timestamp")
        total = len(cdf)
        if total < 3:
            continue

        step = max(2, total // n_windows)
        actual_windows = max(3, min(n_windows, total // step))
        timeline = []
        for i in range(1, actual_windows + 1):
            cutoff = min(i * step, total)
            if cutoff < 2:
                continue
            subset = cdf.iloc[:cutoff]
            sub_total = len(subset)
            sub_accounts = int(subset["user_id"].nunique())

            sync = compute_synchrony(subset["timestamp"])
            age = compute_account_age_clustering(subset["account_created_at"])
            txt_acct = compute_text_account_ratio(sub_total, sub_accounts)
            div = compute_account_diversity(sub_total, sub_accounts)

            if "media_hash" in df.columns:
                sub_hashes = subset["media_hash"].dropna()
                sub_hashes = sub_hashes[sub_hashes != ""]
                if len(sub_hashes) > 0:
                    hc = sub_hashes.value_counts()
                    shared = hc[hc > 1].sum() if not hc.empty else 0
                    intra = shared / sub_total
                    ch_set = set(sub_hashes.unique())
                    cross_shared = sum(1 for h in ch_set if len(hash_to_clusters.get(h, set())) > 1)
                    cross = cross_shared / len(ch_set) if ch_set else 0
                    mr = 0.6 * intra + 0.4 * cross
                else:
                    mr = 0
            else:
                mr = 0

            coord = (
                0.40 * sync
                + 0.35 * txt_acct
                + 0.25 * (1 - div)
            )
            coord = min(max(coord, 0), 1)

            acct_dates = pd.to_datetime(subset["account_created_at"]).dropna()
            if len(acct_dates) > 1:
                age_spread = (acct_dates.max() - acct_dates.min()).days
                age_risk = 1 - min(age_spread / 180, 1)
            else:
                age_risk = 0

            accel = compute_acceleration(subset["timestamp"])

            htri_val = 0.30 * coord + 0.25 * mr + 0.25 * accel + 0.20 * age_risk
            htri_val = min(max(htri_val, 0), 1)

            timeline.append({
                "time": subset["timestamp"].max().isoformat(),
                "htri": round(htri_val, 4),
                "coordination": round(coord, 4),
                "media_reuse": round(mr, 4),
                "acceleration": round(accel, 4),
                "age_risk": round(age_risk, 4),
                "post_count": sub_total,
            })

        results[int(label)] = timeline
    return results
