import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from typing import Optional
from src.coordination import compute_all_signals, compute_acceleration, compute_new_accounts_influx

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def load_model():
    print(f"[NLP] Model yukleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[NLP] Model hazir (output dim: {model.get_embedding_dimension()})")
    return model


def compute_embeddings(
    texts: list[str],
    model: Optional[SentenceTransformer] = None,
    force: bool = False,
) -> np.ndarray:
    cache_file = _cache_path("embeddings.npy")
    if not force and os.path.exists(cache_file):
        emb = np.load(cache_file)
        print(f"[NLP] Embedding cache'ten yuklendi: {emb.shape}")
        return emb

    if model is None:
        model = load_model()
    print(f"[NLP] Embedding hesaplaniyor ({len(texts)} metin)...")
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    np.save(cache_file, emb)
    print(f"[NLP] Embedding kaydedildi: {emb.shape}")
    return emb


def cluster_embeddings(
    embeddings: np.ndarray,
    eps: float = 0.35,
    min_samples: int = 2,
    metric: str = "cosine",
) -> tuple[np.ndarray, int]:
    print(f"[NLP] DBSCAN kumeleme (eps={eps}, min_samples={min_samples})...")
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"[NLP] Kume sayisi: {n_clusters} (+ {n_noise} gurultu)")
    print(f"[NLP] Kume dagilimi: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return labels, n_clusters


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    cache_file = _cache_path("tsne.npy")
    if os.path.exists(cache_file):
        tsne = np.load(cache_file)
        print(f"[NLP] t-SNE cache'ten yuklendi: {tsne.shape}")
        return tsne

    n = embeddings.shape[0]
    actual_perp = min(perplexity, max(1, n // 3))
    print(f"[NLP] t-SNE hesaplaniyor (perplexity={actual_perp})...")
    tsne = TSNE(
        n_components=2,
        perplexity=actual_perp,
        random_state=random_state,
        metric="cosine",
    ).fit_transform(embeddings)
    np.save(cache_file, tsne)
    return tsne


TURKISH_STOPWORDS = {
    "bir", "bu", "ve", "veya", "ile", "icin", "ama", "ancak",
    "cok", "daha", "en", "gibi", "kadar", "sonra", "once",
    "sadece", "veya", "ya", "yada", "hem", "hic", "her",
    "bunu", "buna", "bunun", "siz", "ben", "biz", "sen",
    "onun", "ona", "onu", "onlar", "bize", "bize", "size",
    "daha", "yeni", "var", "yok", "degil", "mi", "mu",
    "tabi", "hatta", "ayrica", "ozellikle", "acaba",
    "sadece", "belki", "yine", "yoksa", "nasil", "ne",
    "neden", "nicin", "kim", "hangi",
}


def compute_tfidf_cluster_labels(df_with_labels: pd.DataFrame) -> dict[int, str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_texts = {}
    for label in sorted(set(df_with_labels["cluster_label"])):
        if label == -1:
            continue
        texts = df_with_labels[df_with_labels["cluster_label"] == label]["text"].tolist()
        cluster_texts[label] = " ".join(texts)

    if not cluster_texts:
        return {}

    labels_sorted = sorted(cluster_texts.keys())
    documents = [cluster_texts[l] for l in labels_sorted]

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words=list(TURKISH_STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        token_pattern=r"(?u)\b\w+\b",
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        result = {}
        for idx, label in enumerate(labels_sorted):
            row = tfidf_matrix[idx].toarray().flatten()
            top_indices = row.argsort()[-5:][::-1]
            top_terms = [feature_names[i] for i in top_indices if row[i] > 0][:3]
            if top_terms:
                combined = " / ".join(t.title() for t in top_terms)
                result[label] = combined[:50]
            else:
                result[label] = f"Kume_{label}"
        return result
    except Exception:
        return {l: f"Kume_{l}" for l in labels_sorted}


def build_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    tsne_coords: Optional[np.ndarray] = None,
) -> list[dict]:
    df = df.copy()
    df["cluster_label"] = labels

    all_signals = compute_all_signals(df, labels)
    tfidf_labels = compute_tfidf_cluster_labels(df)

    summaries = []
    for label in sorted(set(labels)):
        cdf = df[df["cluster_label"] == label]
        ts = pd.to_datetime(cdf["timestamp"])
        total = len(cdf)
        accounts = int(cdf["user_id"].nunique())
        first_seen = ts.min()
        last_seen = ts.max()

        acceleration = compute_acceleration(ts)
        growth_rate = min(total / max((last_seen - first_seen).total_seconds() / 3600, 1) / 5, 1.0)
        new_accounts = compute_new_accounts_influx(ts, cdf["user_id"])

        platform_dist = cdf["source_platform"].value_counts().to_dict()

        acct_dates = pd.to_datetime(cdf["account_created_at"]).dropna()
        if len(acct_dates) > 1:
            age_spread = (acct_dates.max() - acct_dates.min()).days
            age_risk = 1 - min(age_spread / 180, 1)
        else:
            age_risk = 0

        sig = all_signals.get(int(label), {})
        coordination_score = sig.get("coordination_score", 0)
        media_reuse_score = sig.get("media_reuse_score", 0)
        synchrony = sig.get("synchrony", 0)
        acct_diversity = sig.get("account_diversity", 0)
        coord_sim = sig.get("text_account_ratio", 0)

        htri = (
            0.30 * coordination_score
            + 0.25 * media_reuse_score
            + 0.25 * acceleration
            + 0.20 * age_risk
        )
        htri = min(max(htri, 0), 1)

        if htri >= 0.80:
            risk = "Kritik"
        elif htri >= 0.60:
            risk = "Yuksek"
        elif htri >= 0.30:
            risk = "Orta"
        else:
            risk = "Dusuk"

        tsne_x, tsne_y = None, None
        if tsne_coords is not None:
            mask = df["cluster_label"] == label
            tsne_x = tsne_coords[mask, 0].tolist()
            tsne_y = tsne_coords[mask, 1].tolist()

        label_str = tfidf_labels.get(int(label), f"Kume_{label}") if label >= 0 else "Gurultu"

        summaries.append({
            "cluster_id": int(label),
            "label": label_str,
            "is_noise": label == -1,
            "post_count": total,
            "unique_accounts": accounts,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "growth_rate": round(growth_rate, 4),
            "acceleration": round(acceleration, 4),
            "new_accounts": new_accounts,
            "platforms": platform_dist,
            "synchrony": round(synchrony, 4),
            "age_risk": round(age_risk, 4),
            "coord_sim": round(coord_sim, 4),
            "acct_diversity": round(acct_diversity, 4),
            "coordination_score": round(coordination_score, 4),
            "media_reuse_score": round(media_reuse_score, 4),
            "media_intra_overlap": sig.get("media_intra_overlap", 0),
            "media_cross_overlap": sig.get("media_cross_overlap", 0),
            "htri": round(htri, 4),
            "risk_level": risk,
            "tsne_x": tsne_x,
            "tsne_y": tsne_y,
            "size": total,
        })

    summaries.sort(key=lambda s: s["post_count"], reverse=True)
    return summaries


def run_pipeline(
    df: pd.DataFrame,
    eps: float = 0.35,
    min_samples: int = 2,
    force_embed: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    model = load_model()
    texts = df["text"].tolist()
    embeddings = compute_embeddings(texts, model, force=force_embed)
    labels, n_clusters = cluster_embeddings(embeddings, eps=eps, min_samples=min_samples)
    tsne = compute_tsne(embeddings, perplexity=min(30, len(df) - 1))
    summaries = build_cluster_summary(df, labels, tsne_coords=tsne)
    return labels, tsne, summaries


def query_narrative(
    text: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    summaries: list[dict],
    threshold: float = 0.3,
) -> dict:
    texts = df["text"].tolist()
    query_emb = model.encode([text], normalize_embeddings=True)[0]

    centroids = {}
    for s in summaries:
        if s["is_noise"]:
            continue
        cid = s["cluster_id"]
        mask = labels == cid
        centroids[cid] = embeddings[mask].mean(axis=0)

    best_cid, best_sim = None, -1.0
    for cid, cent in centroids.items():
        sim = float(np.dot(query_emb, cent))
        if sim > best_sim:
            best_sim = sim
            best_cid = cid

    if best_cid is None or best_sim < threshold:
        return {"found": False, "similarity": best_sim if best_cid else 0}

    cluster_summary = next(s for s in summaries if s["cluster_id"] == best_cid)

    similar_texts = []
    for _, row in df[labels == best_cid].iterrows():
        similar_texts.append({
            "text": row["text"][:120],
            "platform": row["source_platform"],
            "timestamp": row["timestamp"].isoformat(),
        })

    return {
        "found": True,
        "similarity": round(best_sim, 4),
        "cluster": cluster_summary,
        "similar_texts": similar_texts[:5],
    }
