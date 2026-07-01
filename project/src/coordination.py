import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from src.media_reuse import compute_media_reuse


def compute_synchrony(timestamps: pd.Series) -> float:
    """Sinyal 1 — Es zamanlilik: Paylasimlar ne kadar dar pencerede."""
    if len(timestamps) < 2:
        return 0.0
    ts = pd.to_datetime(timestamps).sort_values()
    diffs = ts.diff().dt.total_seconds().dropna()
    if len(diffs) == 0:
        return 1.0
    d_mean = diffs.mean()
    if d_mean == 0:
        return 1.0
    d_std = diffs.std()
    if pd.isna(d_std) or d_std == 0:
        return 1.0
    cv = d_std / max(d_mean, 1)
    return float(1 - min(cv / 2, 1))


def compute_account_age_clustering(account_dates: pd.Series) -> float:
    """Sinyal 2 — Hesap yasi kumelenmesi: Hesaplar ne kadar yakin tarihte acilmis."""
    dates = pd.to_datetime(account_dates).dropna()
    if len(dates) < 2:
        return 0.0
    spread = (dates.max() - dates.min()).days
    return float(1 - min(spread / 180, 1))


def compute_text_account_ratio(total_posts: int, unique_accounts: int) -> float:
    """Sinyal 3 — Metin benzerligi / hesap orani: Kac farkli hesaptan kac post."""
    if total_posts == 0:
        return 0.0
    return float(max(0, 1 - unique_accounts / max(total_posts, 1)))


def compute_account_diversity(total_posts: int, unique_accounts: int) -> float:
    """Sinyal 4 — Hesap cesitliligi (dusuk = risk): unique / total."""
    if total_posts == 0:
        return 0.0
    return float(unique_accounts / total_posts)


def compute_coordination_score(
    synchrony: float,
    text_account_ratio: float,
    account_diversity: float,
) -> float:
    """3 davranissal sinyalin agirlikli bilesimi -> 0-1 arasi koordinasyon skoru.
    Medya Tekrari ve Hesap Yasi burada degil, sadece HTRI'de bagimsiz bilesen olarak yer alir."""
    score = (
        0.40 * synchrony
        + 0.35 * text_account_ratio
        + 0.25 * (1 - account_diversity)
    )
    return float(min(max(score, 0), 1))


def compute_all_signals(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """Her kume icin 5 sinyali ve koordinasyon skorunu hesaplar."""
    media_reuse_results = compute_media_reuse(df, labels)

    results = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        cdf = df[labels == label]
        total = len(cdf)
        unique_accounts = int(cdf["user_id"].nunique())

        synchrony = compute_synchrony(cdf["timestamp"])
        age_clustering = compute_account_age_clustering(cdf["account_created_at"])
        text_account_ratio = compute_text_account_ratio(total, unique_accounts)
        account_diversity = compute_account_diversity(total, unique_accounts)
        reuse = media_reuse_results.get(int(label), {}).get("media_reuse_score", 0)
        intra = media_reuse_results.get(int(label), {}).get("intra_overlap", 0)
        cross = media_reuse_results.get(int(label), {}).get("cross_overlap", 0)

        coordination = compute_coordination_score(
            synchrony, text_account_ratio, account_diversity
        )

        results[int(label)] = {
            "synchrony": round(synchrony, 4),
            "age_clustering": round(age_clustering, 4),
            "text_account_ratio": round(text_account_ratio, 4),
            "account_diversity": round(account_diversity, 4),
            "media_reuse_score": round(reuse, 4),
            "media_intra_overlap": round(intra, 4),
            "media_cross_overlap": round(cross, 4),
            "coordination_score": round(coordination, 4),
            "total_posts": total,
            "unique_accounts": unique_accounts,
        }
    return results


def build_interaction_network(
    df: pd.DataFrame, labels: np.ndarray, min_edge_weight: int = 2
) -> nx.Graph:
    """Kume bazli ortak hesap iliskilerini cikarir."""
    G = nx.Graph()
    cluster_accounts = {}
    for label in set(labels):
        if label == -1:
            continue
        accounts = set(df[labels == label]["user_id"].unique())
        cluster_accounts[int(label)] = accounts
        G.add_node(int(label), size=len(accounts))

    labels_list = list(cluster_accounts.keys())
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            shared = cluster_accounts[labels_list[i]] & cluster_accounts[labels_list[j]]
            if len(shared) >= min_edge_weight:
                G.add_edge(labels_list[i], labels_list[j], weight=len(shared))
    return G


def compute_acceleration(timestamps: pd.Series, min_posts: int = 5) -> float:
    ts = pd.to_datetime(timestamps).sort_values()
    n = len(ts)
    if n < min_posts:
        return 0.0
    t_min, t_max = ts.min(), ts.max()
    total_hours = max((t_max - t_min).total_seconds() / 3600, 1)
    n_windows = max(2, min(4, n // 5))
    window_hours = total_hours / n_windows
    rates = []
    for i in range(n_windows):
        w_start = t_min + pd.Timedelta(hours=i * window_hours)
        w_end = t_min + pd.Timedelta(hours=(i + 1) * window_hours)
        count = max(((ts >= w_start) & (ts < w_end)).sum(), 1)
        rates.append(count / window_hours)

    if len(rates) < 2:
        return 0.0

    changes = []
    for i in range(1, len(rates)):
        if rates[i - 1] > 0:
            rel_change = (rates[i] - rates[i - 1]) / rates[i - 1]
            changes.append(max(rel_change, -1))

    if not changes:
        return 0.0

    max_change = max(changes)
    return float(min(max(max_change / 2, 0), 1))


def compute_new_accounts_influx(timestamps: pd.Series, user_ids: pd.Series) -> int:
    ts = pd.to_datetime(timestamps)
    df = pd.DataFrame({"ts": ts, "uid": user_ids})
    df = df.sort_values("ts")
    n = len(df)
    half = n // 2
    if half == 0:
        return 0
    first_accts = set(df.iloc[:half]["uid"])
    second_accts = set(df.iloc[half:]["uid"])
    return len(second_accts - first_accts)
