import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import networkx as nx
import os
import io
# import warnings  # (Faz 10 icindi, geri alindi)

st.set_page_config(page_title="FIMI-TR | Erken Uyarı Platformu", layout="wide")

st.title("FIMI-TR")
st.subheader("Türkçe Bilgi Manipülasyonu Erken Uyarı Platformu")

st.markdown("""
<style>
    html, body, [class*="css"] {font-size:16px;}
    .stMetric label, .stMetric [data-testid="stMetricLabel"] {font-size:14px!important;}
    .stMetric [data-testid="stMetricValue"] {font-size:28px!important;}
    .st-expander header {font-size:15px!important;}
    button[data-baseweb="tab"] {font-size:15px!important;padding:10px 18px!important;}
    .stTabs [data-baseweb="tab-list"] {gap:4px;}
    .stSelectbox, .stMultiSelect {font-size:14px!important;}
    @media (max-width: 768px) {
        .stMetric [data-testid="stMetricValue"] {font-size:22px!important;}
        button[data-baseweb="tab"] {font-size:13px!important;padding:6px 10px!important;}
    }
</style>
""", unsafe_allow_html=True)

# ── P0.4: Renk Lejantı (tek sefer, en üstte) ──
st.markdown(
    '<div style="display:flex;gap:16px;font-size:13px;margin-bottom:8px;flex-wrap:wrap;">'
    '<span><span style="color:#D55E00;">●</span> Kritik</span>'
    '<span><span style="color:#E69F00;">●</span> Yuksek</span>'
    '<span><span style="color:#56B4E9;">●</span> Orta</span>'
    '<span><span style="color:#009E73;">●</span> Dusuk</span>'
    '</div>',
    unsafe_allow_html=True,
)

RISK_COLORS = {"Kritik": "#D55E00", "Yuksek": "#E69F00", "Orta": "#56B4E9", "Dusuk": "#009E73"}
RISK_ORDER = {"Kritik": 0, "Yuksek": 1, "Orta": 2, "Dusuk": 3}

# ── Karşılama metni ──
st.markdown(
    '<div style="background:#f0f4ff;border-left:4px solid #4A6FE0;border-radius:8px;padding:14px 18px;margin-bottom:12px;font-size:14px;line-height:1.5;color:#1a1a2e;">'
    "Bu platform, T\u00fcrk\u00e7e haber ve sosyal medya i\u00e7eriklerindeki bilgi manip\u00fclasyonu "
    "(dezenformasyon) giri\u015fimlerini erken tespit etmek i\u00e7in yapay zeka kullan\u0131r. "
    "A\u015fa\u011f\u0131ya \u015f\u00fcpheli bir haber metni yap\u0131\u015ft\u0131rarak analiz edebilir "
    "veya mevcut uyar\u0131lar\u0131 inceleyebilirsiniz."
    '</div>',
    unsafe_allow_html=True,
)

def _signal_bar(val: float) -> str:
    color = "#E74C3C" if val > 0.6 else "#F39C12" if val > 0.3 else "#2ECC71"
    pct = min(int(val * 100), 100)
    return f'<div style="display:flex;align-items:center;gap:6px;"><div style="width:60px;height:6px;background:#eee;border-radius:3px;overflow:hidden;"><div style="width:{pct}%;height:100%;background:{color};border-radius:3px;"></div></div><span style="font-size:12px;color:#555;">{val:.2f}</span></div>'

def _risk_badge(level: str) -> str:
    c = RISK_COLORS.get(level, "#95A5A6")
    return f'<span style="background:{c};color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:bold;">{level}</span>'

def _cluster_matches_filters(s, df, labels, risk_levels, platforms, date_range):
    if s["is_noise"]: return False
    if s["risk_level"] not in risk_levels:
        return False
    cdf = df[labels == s["cluster_id"]]
    if platforms:
        if not cdf["source_platform"].isin(platforms).any():
            return False
    if date_range and len(date_range) == 2:
        t_min, t_max = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + timedelta(days=1)
        if not ((cdf["timestamp"] >= t_min) & (cdf["timestamp"] < t_max)).any():
            return False
    return True

# ──────────────────────────────────────────────
# VERI YUKLEME
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("data/synthetic_data.csv", parse_dates=["timestamp", "account_created_at"])
    df["repost_of"] = df["repost_of"].fillna("").astype(str)
    df["mentions"] = df["mentions"].fillna("").astype(str)
    return df

@st.cache_data
def run_nlp_pipeline(_df):
    from src.nlp_pipeline import run_pipeline
    labels, tsne, summaries = run_pipeline(_df, eps=0.45, min_samples=2)
    return labels.tolist(), tsne.tolist(), summaries

@st.cache_data
def compute_evolutions(_df, _labels):
    from src.narrative_evolution import compute_all_evolutions
    labels_arr = np.array(_labels)
    return compute_all_evolutions(_df, labels_arr)

@st.cache_data
def compute_htri_timeline_cached(_df, _labels):
    from src.htri_timeline import compute_cluster_htri_timeline
    return compute_cluster_htri_timeline(_df, np.array(_labels))

@st.cache_resource
def get_model():
    from src.nlp_pipeline import load_model
    return load_model()

@st.cache_data
def get_embeddings_cached(_df):
    from src.nlp_pipeline import compute_embeddings, load_model
    model = load_model()
    texts = _df["text"].tolist()
    return compute_embeddings(texts, model).tolist()

df = load_data()
labels_list, tsne_list, summaries = run_nlp_pipeline(df)
labels = np.array(labels_list)
tsne = np.array(tsne_list)

df["cluster_id"] = labels

evolutions = compute_evolutions(df, labels_list)
htri_timeline = compute_htri_timeline_cached(df, labels_list)

all_accounts = df["user_id"].nunique()
all_platforms = sorted(df["source_platform"].unique().tolist())
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

# ──────────────────────────────────────────────
# SIDEBAR FILTRELER
# ──────────────────────────────────────────────

with st.sidebar:
    with st.expander("Filtreler (opsiyonel)", expanded=True):
        st.caption("Asagidaki kriterlere gore anlatilari filtreleyin.")

        selected_risk = st.multiselect(
            "Risk Seviyesi",
            ["Kritik", "Yuksek", "Orta", "Dusuk"],
            default=["Kritik", "Yuksek", "Orta", "Dusuk"],
            help="Sadece sectiginiz risk seviyesindeki anlatilari goster",
        )

        selected_platforms = st.multiselect(
            "Platform",
            all_platforms,
            default=all_platforms,
            help="Sadece sectiginiz platformlardaki icerikleri goster",
        )

        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("Baslangic", min_date, key="filter_date_from")
        with col2:
            date_to = st.date_input("Bitis", max_date, key="filter_date_to")

        date_range = (date_from, date_to) if date_from and date_to else None

        st.caption("Filtreler tum sekmelere uygulanir.")

# ──────────────────────────────────────────────
# FILTRELENMIS VERI
# ──────────────────────────────────────────────

filtered_summaries = [
    s for s in summaries
    if _cluster_matches_filters(s, df, labels, selected_risk, selected_platforms, date_range)
]

n_critical = sum(1 for s in filtered_summaries if s["risk_level"] == "Kritik")
n_high = sum(1 for s in filtered_summaries if s["risk_level"] == "Yuksek")
n_medium = sum(1 for s in filtered_summaries if s["risk_level"] == "Orta")
n_low = sum(1 for s in filtered_summaries if s["risk_level"] == "Dusuk")

active_narratives = len([s for s in filtered_summaries if s["post_count"] >= 3])

df_visible = df[df["cluster_id"].isin(s["cluster_id"] for s in filtered_summaries)]

# ── P0.2: Durum Özeti Banner (dinamik, tek satır) ──
if n_critical > 0:
    banner_color, banner_icon = "#E74C3C", "&#9888;"
    banner_text = f"UYARI: {n_critical} kritik, {n_high} yuksek riskli anlati tespit edildi."
elif n_high > 0:
    banner_color, banner_icon = "#E67E22", "&#9888;"
    banner_text = f"DIKKAT: {n_high} yuksek riskli anlati mevcut."
else:
    banner_color, banner_icon = "#2ECC71", "&#10003;"
    banner_text = "Su an icin kritik veya yuksek riskli anlati bulunmuyor."
st.markdown(
    f'<div style="background:{banner_color}15;border:1px solid {banner_color}30;border-radius:8px;padding:8px 14px;font-size:14px;font-weight:600;color:{banner_color};">'
    f'<span style="margin-right:8px;">{banner_icon}</span>{banner_text}</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# SUMMARY CARDS
# ──────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Toplam Icerik", f"{len(df_visible):,}", help="Filtrelenen toplam gonderi sayisi")
col2.metric("Aktif Anlati", active_narratives, help="En az 3 gonderisi olan anlati kumeleri")
col3.metric(
    f"Kritik Risk",
    n_critical,
    delta=None,
    help=f"{n_critical} anlati en yuksek risk seviyesinde",
)
col4.metric(
    f"Yuksek Risk",
    n_high,
    delta=None,
    help=f"{n_high} anlati yuksek risk seviyesinde",
)
col5.metric(
    f"Orta / Dusuk",
    n_medium + n_low,
    delta=None,
    help=f"{n_medium} orta + {n_low} dusuk riskli anlati",
)

# ── P0.2: Haber Sorgula ──
st.markdown("---")
st.subheader("🔍 Haber Sorgula")
st.markdown(
    "Ş\u00fcpheli bir haber metnini a\u015fa\u011f\u0131ya yap\u0131\u015ft\u0131r\u0131n, "
    "sistem en yak\u0131n anlat\u0131 grubunu bularak risk de\u011ferlendirmesi yaps\u0131n. "
    '<span style="color:#888;font-size:12px;cursor:help;" title="Metin yapay zeka ile benzer iceriklere gore analiz edilir.">'
    "\u24d8</span>",
    unsafe_allow_html=True,
)

query_text = st.text_area(
    "Haber metni",
    placeholder="Haber metnini veya sosyal medya g\u00f6nderisini buraya yap\u0131\u015ft\u0131r\u0131n...",
    height=90,
    label_visibility="collapsed",
)
query_clicked = st.button("\U0001F50D Analiz Et", type="primary")

if query_clicked and query_text.strip():
    with st.spinner("Analiz ediliyor..."):
        model = get_model()
        embeddings_arr = np.array(get_embeddings_cached(df))
        from src.nlp_pipeline import query_narrative

        qresult = query_narrative(
            text=query_text.strip(),
            model=model,
            embeddings=embeddings_arr,
            labels=labels,
            df=df,
            summaries=summaries,
        )

        if qresult["found"]:
            s = qresult["cluster"]
            rc = RISK_COLORS.get(s["risk_level"], "#888")
            sim_pct = int(qresult["similarity"] * 100)

            score_col, badge_col = st.columns([1, 1])
            with score_col:
                st.metric("Risk Skoru", f"{s['htri']*100:.0f}/100",
                          help=f"0-100 arasi risk puani. Yüksek = daha tehlikeli.")
            with badge_col:
                st.markdown(f"<div style='margin-top:24px;'>{_risk_badge(s['risk_level'])}</div>",
                            unsafe_allow_html=True)

            st.markdown(f"**{s['label'][:45]}**")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Benzerlik", f"%{sim_pct}", help="Girilen metnin bu anlati grubuyla benzerlik orani")
            m2.metric("Gonderi", s["post_count"], help="Bu gruptaki toplam gonderi sayisi")
            m3.metric("Hesap", s["unique_accounts"], help="Bu gruptaki farkli hesap sayisi")
            m4.metric("Yayilma Hizi", f"%{min(int(s.get('acceleration', 0) * 100), 100)}",
                      help="Gonderilerin yayilma ivmesi")

            if qresult["similar_texts"]:
                with st.expander("Benzer icerikler"):
                    for item in qresult["similar_texts"]:
                        st.markdown(
                            f'<div style="font-size:12px;padding:4px 0;border-bottom:1px solid #eee;">'
                            f'<span style="color:#888;">[{item["platform"]}]</span> {item["text"]}</div>',
                            unsafe_allow_html=True,
                        )
        else:
            sim_pct = int(qresult["similarity"] * 100)
            st.info(
                f"Bu metin daha \u00f6nce g\u00f6r\u00fclmedi (benzerlik: %{sim_pct}). "
                "Mevcut anlat\u0131 gruplar\u0131yla e\u015fle\u015fmedi."
            )

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────

tab0, tab1, tab2, tab3 = st.tabs([
    "Yonetim Paneli",
    "Anlati Gruplari",
    "Organizasyon Analizi",
    "Dogrulama Bulgulari",
])

# ═══════════════════════════════════════════════
# TAB 0 — Executive Dashboard
# ═══════════════════════════════════════════════

with tab0:
    st.subheader("Alarm Merkezi")
    st.caption("Kritik ve yuksek riskli anlatilarin gercek zamanli durumu, risk skoru zaman serisi ve risk dagilimi.")

    critical = [s for s in filtered_summaries if s["risk_level"] == "Kritik" and s["post_count"] >= 3]
    if critical:
        st.markdown("### Yuksek Oncelikli Alarmlar")
        ccols = st.columns(min(len(critical), 4))
        for i, s in enumerate(critical[:4]):
            cid = s["cluster_id"]
            tl = htri_timeline.get(cid, [])
            recent_change = ""
            if len(tl) >= 3:
                last_new = tl[-1]["post_count"] - tl[-2]["post_count"]
                prev_new = tl[-2]["post_count"] - tl[-3]["post_count"]
                if prev_new > 0:
                    pct = int((last_new - prev_new) / prev_new * 100)
                    if pct > 0:
                        recent_change = f"Son Pencere: +{pct}%"
            accel_pct = min(int(s.get("acceleration", 0) * 100), 100)
            card_html = f"""
            <div style="border:1px solid #E74C3C44; border-radius:10px; padding:14px; background:#fff5f5; border-left:4px solid #E74C3C;">
                <div style="font-size:32px; font-weight:bold; color:#E74C3C;">{s['htri']:.2f}</div>
                <div style="font-size:13px; font-weight:bold; margin:6px 0 4px;">{s['label'][:30]}</div>
                <div style="font-size:11px; color:#666;">
                    Ivme: <b>{accel_pct}%</b> &nbsp;|&nbsp; Yeni: <b>+{s.get('new_accounts', 0)}</b> hesap
                </div>
                <div style="font-size:12px; color:#E74C3C; font-weight:bold; margin-top:4px;">{recent_change}</div>
            </div>
            """
            with ccols[i]:
                st.markdown(card_html, unsafe_allow_html=True)

    alert_clusters = sorted(
        [s for s in filtered_summaries if s["risk_level"] in ("Kritik", "Yuksek") and s["post_count"] >= 3],
        key=lambda x: (RISK_ORDER.get(x["risk_level"], 99), -x["htri"]),
    )

    if not alert_clusters:
        st.success("Su an icin kritik veya yuksek riskli anlati bulunmuyor.")
    else:
        st.markdown(f"**{len(alert_clusters)} anlati yakindan izlenmeli**")

        card_cols = st.columns(2)
        for idx, s in enumerate(alert_clusters[:10]):
            cid = s["cluster_id"]
            tl = htri_timeline.get(cid, [])
            recent_new = ""
            if len(tl) >= 3:
                last_new = tl[-1]["post_count"] - tl[-2]["post_count"]
                prev_new = tl[-2]["post_count"] - tl[-3]["post_count"]
                if prev_new > 0:
                    pct = int((last_new - prev_new) / prev_new * 100)
                    if pct > 0:
                        recent_new = f"+{pct}%"
            risk_c = RISK_COLORS.get(s["risk_level"], "#E74C3C")
            with card_cols[idx % 2]:
                card = f"""
                <div style="border:1px solid {risk_c}44;border-radius:10px;padding:12px;background:{risk_c}08;border-left:4px solid {risk_c};margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:28px;font-weight:bold;color:{risk_c};">{s['htri']:.2f}</span>
                        {_risk_badge(s['risk_level'])}
                    </div>
                    <div style="font-size:13px;font-weight:bold;margin:6px 0 4px;">{s['label'][:35]}</div>
                    <div style="font-size:11px;color:#666;">
                        Gonderi: <b>{s['post_count']}</b> &nbsp;|&nbsp;
                        Hesap: <b>{s['unique_accounts']}</b>
                    </div>
                    <div style="font-size:11px;color:#666;margin-top:2px;">
                        Ivme: <b>{min(int(s.get('acceleration', 0) * 100), 100)}%</b> &nbsp;|&nbsp;
                        Organizasyon: <b>{s['coordination_score']:.2f}</b> &nbsp;|&nbsp;
                        Son: <b>{s['last_seen'].strftime('%d.%m %H:%M')}</b>
                    </div>
                    {f'<div style="font-size:12px;color:{risk_c};font-weight:bold;margin-top:4px;">Son Pencere: {recent_new}</div>' if recent_new else ''}
                </div>
                """
                st.markdown(card, unsafe_allow_html=True)
                with st.expander(f"Detay — {s['label'][:25]}", expanded=False):
                    c1d, c2d = st.columns(2)
                    with c1d:
                        st.markdown(f"**Risk Skoru:** {s['htri']:.4f}")
                        st.markdown(f"**Organizasyon:** {s['coordination_score']:.4f}")
                        st.markdown(f"**Medya Tekrar:** {s['media_reuse_score']:.4f}")
                        st.markdown(f"**Yayilma Hizi:** {s.get('acceleration', 0):.4f}")
                    with c2d:
                        st.markdown(f"**Hesap Yasi Riski:** {s.get('age_risk', 0):.4f}")
                        st.markdown(f"**Yayilma Hizi:** {s['growth_rate']:.4f}")
                        st.markdown(f"**Yeni Hesap:** {s.get('new_accounts', 0)}")
                        st.markdown(f"**Ilk Gorulme:** {s['first_seen'].strftime('%d.%m.%Y %H:%M')}")
                    share_text = f"FIMI-TR Uyarisi: {s['label'][:40]} | Risk: {s['risk_level']} | Skor: {s['htri']:.2f}"
                    st.markdown(f"**Paylas:**")
                    st.code(share_text, language="text")

    col_a, col_b = st.columns(2)

    # ─── HTRI Time Series ───
    with col_a:
        st.markdown("---")
        st.subheader("Risk Skoru Zaman Serisi")
        st.caption("Her anlatinin zaman icinde risk skoru degisimi. Cizgi yukseliyorsa tehdit seviyesi artiyor. Sadece kritik/yuksek riskli anlatilar.")

        ts_clusters = [s for s in alert_clusters if htri_timeline.get(s["cluster_id"], [])]
        fig_ts = go.Figure()
        for s in ts_clusters:
            cid = s["cluster_id"]
            tl = htri_timeline[cid]
            fig_ts.add_trace(go.Scatter(
                x=[t["time"] for t in tl],
                y=[t["htri"] for t in tl],
                mode="lines+markers",
                name=s["label"][:25],
                line=dict(color=RISK_COLORS.get(s["risk_level"], "#888"), width=2),
                hovertemplate="<b>%{text}</b><br>Risk Skoru: %{y:.2f}<br>Post: %{customdata}<extra></extra>",
                text=[s["label"][:30]] * len(tl),
                customdata=[t["post_count"] for t in tl],
            ))
        if ts_clusters:
            fig_ts.update_layout(
                height=350, margin=dict(t=10, b=10, l=10, r=10),
                xaxis_title="Zaman", yaxis_title="Risk Skoru",
                yaxis=dict(range=[0, 1]),
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_ts, width="stretch", key="htri_timeseries")
        else:
            st.info("Zaman serisi icin yeterli veri yok.")

    # ─── Post Volume Timeline ───
    with col_b:
        st.markdown("---")
        st.subheader("Gonderi Hacmi Zamani")
        st.caption("Zaman icinde her anlatinin gonderi hacmi. Ani yukselisler koordineli kampanya isareti olabilir.")

        volume_data = []
        for s in filtered_summaries:
            if s["post_count"] < 3:
                continue
            cid = s["cluster_id"]
            tl = htri_timeline.get(cid, [])
            if len(tl) >= 2:
                prev_count = 0
                for t in tl:
                    volume_data.append({
                        "Zaman": t["time"],
                        "Gonderi Sayisi": t["post_count"] - prev_count,
                        "Anlati": s["label"][:25],
                        "Risk": s["risk_level"],
                    })
                    prev_count = t["post_count"]

        if volume_data:
            df_vol = pd.DataFrame(volume_data)
            fig_vol = px.area(
                df_vol, x="Zaman", y="Gonderi Sayisi", color="Anlati",
                color_discrete_map={s["label"][:25]: RISK_COLORS.get(s["risk_level"], "#888") for s in filtered_summaries},
                groupnorm=None,
            )
            fig_vol.update_layout(
                height=350, margin=dict(t=10, b=10, l=10, r=10),
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_vol, width="stretch", key="volume_timeline")
        else:
            st.info("Hacim verisi icin yeterli bilgi yok.")

    # ─── Export ───
    st.markdown("---")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        if filtered_summaries:
            df_export = pd.DataFrame([{
                "Anlati": s["label"],
                "Risk": s["risk_level"],
                "Risk Skoru": s["htri"],
                "Organizasyon": s["coordination_score"],
                "Medya Tekrar": s["media_reuse_score"],
                "Gonderi": s["post_count"],
                "Hesap": s["unique_accounts"],
                "Yayilma Hizi": s["growth_rate"],
                "Ivme": s.get("acceleration", 0),
                "Ilk Gorulme": s["first_seen"].isoformat(),
                "Son Gorulme": s["last_seen"].isoformat(),
            } for s in filtered_summaries if s["post_count"] >= 3])
            csv_buf = io.StringIO()
            df_export.to_csv(csv_buf, index=False)
            st.download_button(
                "Anlati Raporu (CSV)",
                data=csv_buf.getvalue(),
                file_name=f"fimi_tr_rapor_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Filtrelenmis anlati verilerini CSV olarak indir",
            )
    with col_exp2:
        csv_all = io.StringIO()
        df_visible.to_csv(csv_all, index=False)
        st.download_button(
            "Ham Veri (CSV)",
            data=csv_all.getvalue(),
            file_name=f"fimi_tr_ham_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Filtrelenmis ham gonderi verilerini CSV olarak indir",
        )
    with col_exp3:
        if alert_clusters:
            df_alert = pd.DataFrame([{
                "Anlati": s["label"],
                "Risk": s["risk_level"],
                "Risk Skoru": s["htri"],
                "Organizasyon": s["coordination_score"],
                "Gonderi": s["post_count"],
                "Hesap": s["unique_accounts"],
                "Son Gorulme": s["last_seen"].isoformat(),
            } for s in alert_clusters])
            csv_alert = io.StringIO()
            df_alert.to_csv(csv_alert, index=False)
            st.download_button(
                "Alarm Listesi (CSV)",
                data=csv_alert.getvalue(),
                file_name=f"fimi_tr_alarm_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Kritik/yuksek riskli anlatilari CSV olarak indir",
            )

    # ─── Risk Distribution Pie ───
    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        risk_counts = {
            "Kritik": n_critical, "Yuksek": n_high,
            "Orta": n_medium, "Dusuk": n_low,
        }
        risk_df = pd.DataFrame({
            "Risk": list(risk_counts.keys()),
            "Anlati Sayisi": list(risk_counts.values()),
        })
        fig_pie = px.pie(
            risk_df, values="Anlati Sayisi", names="Risk",
            color="Risk", color_discrete_map=RISK_COLORS,
            title="Anlatilarin Risk Dagilimi",
        )
        fig_pie.update_layout(height=280, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, width="stretch", key="risk_dist_pie")
    with col_d:
        platform_counts = df_visible["source_platform"].value_counts().reset_index()
        platform_counts.columns = ["Platform", "Gonderi Sayisi"]
        fig_plat = px.bar(
            platform_counts, x="Platform", y="Gonderi Sayisi",
            color="Platform", text_auto=True,
            title="Platform Bazinda Gonderi Dagilimi",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_plat.update_layout(height=280, margin=dict(t=30, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_plat, width="stretch", key="platform_dist_bar")

# ═══════════════════════════════════════════════
# TAB 1 — Anlati Analizi
# ═══════════════════════════════════════════════

with tab1:
    st.subheader("Anlati Gruplama Sonuclari")
    st.caption("Benzerlik haritasi: metin benzerligine gore gruplanmis anlatilar (ayni renk = ayni grup). Alt kisimda her grubun detayli metrikleri: gonderi sayisi, hesap, platform dagilimi, evrim ve risk skoru.")

    tsne_df = pd.DataFrame({
        "x": tsne[:, 0], "y": tsne[:, 1],
        "cluster_id": [f"Grup_{l}" if l >= 0 else "Diger" for l in labels],
        "text_preview": df["text"].str[:50].tolist(),
        "platform": df["source_platform"].tolist(),
    })
    fig_tsne = px.scatter(
        tsne_df, x="x", y="y", color="cluster_id",
        hover_data={"text_preview": True, "platform": True, "x": False, "y": False},
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={"cluster_id": "Grup"},
        title="Anlati Benzerlik Haritasi — Metin Benzerligine Gore Dagitim",
    )
    fig_tsne.update_traces(marker=dict(size=6))
    fig_tsne.update_layout(height=500, margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig_tsne, width="stretch", key="tsne_proj")

    st.markdown("---")

    display = sorted(
        [s for s in filtered_summaries if s["post_count"] >= 3],
        key=lambda x: x["htri"], reverse=True,
    )[:10]

    if not display:
        st.info("Filtrelere uygun anlati bulunamadi.")
    else:
        for s in display:
            risk_tag = f"[{s['risk_level']}]"
            with st.expander(
                f"{risk_tag} {s['label']} — "
                f"Risk Skoru: {s['htri']:.2f} | "
                f"{s['post_count']} gonderi, {s['unique_accounts']} hesap",
                expanded=False,
            ):
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.markdown(f"**Gonderi Sayisi:** {s['post_count']}")
                    st.markdown(f"**Benzersiz Hesap:** {s['unique_accounts']}")
                    st.markdown(f"**Ilk Gorulme:** {s['first_seen'].strftime('%d.%m.%Y %H:%M')}")
                    st.markdown(f"**Son Gorulme:** {s['last_seen'].strftime('%d.%m.%Y %H:%M')}")
                    st.markdown(f"**Organizasyon Skoru:** {s['coordination_score']:.2f}")
                    st.markdown(f"**Risk Skoru:** {s['htri']:.2f}")
                    st.markdown(f"**Yayilma Hizi:** {s['growth_rate']:.2f}")
                    st.markdown(f"**Ivme:** {s.get('acceleration', 0):.2f}")
                    st.markdown(f"**Yeni Hesap:** {s.get('new_accounts', 0)}")

                with c2:
                    platforms = s["platforms"]
                    if platforms:
                        fig_pie = px.pie(
                            values=list(platforms.values()),
                            names=list(platforms.keys()),
                            title="Platform Dagilimi",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                        )
                        fig_pie.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, width="stretch", key=f"pie_{s['cluster_id']}")

                cid = s["cluster_id"]
                evo = evolutions.get(cid, [])
                if evo:
                    st.markdown("**Anlati Evrimi:**")
                    evo_df = pd.DataFrame(evo)
                    max_change = max(evo_df["change_score"]) if not evo_df.empty else 1

                    for _, row in evo_df.iterrows():
                        bar_pct = min(row["change_score"] / max(max_change, 0.01), 1)
                        bar_color = "#E74C3C" if bar_pct > 0.6 else "#F39C12" if bar_pct > 0.3 else "#2ECC71"
                        bar_html = (
                            f'<div style="display:flex;align-items:center;gap:8px;">'
                            f'<div style="width:100px;height:6px;background:#eee;border-radius:3px;overflow:hidden;">'
                            f'<div style="width:{bar_pct*100:.0f}%;height:100%;background:{bar_color};border-radius:3px;"></div>'
                            f'</div>'
                            f'<span style="font-size:11px;color:#888;">{row["change_score"]:.2f}</span>'
                            f'</div>'
                        )

                        cols = st.columns([1.5, 5, 2])
                        with cols[0]:
                            st.markdown(f'<span style="font-size:13px;color:#555;">{row["time"]}</span>', unsafe_allow_html=True)
                        with cols[1]:
                            direction = "degisim yuksek" if bar_pct > 0.6 else "hafif degisim" if bar_pct > 0.3 else "benzer"
                            icon = "!" if bar_pct > 0.6 else "~" if bar_pct > 0.3 else "."
                            st.markdown(
                                f'<span style="font-size:13px;">{row["text"]}</span>'
                                f'<br><span style="font-size:11px;color:#999;">{icon} {direction}</span>',
                                unsafe_allow_html=True,
                            )
                        with cols[2]:
                            st.markdown(bar_html, unsafe_allow_html=True)

                st.divider()

    # ─── HTRI comparison ───
    st.subheader("Anlati Risk Karsilastirmasi")
    chart_data = [s for s in filtered_summaries if s["post_count"] >= 3]
    if chart_data:
        df_htri = pd.DataFrame(chart_data)
        fig_bar = px.bar(
            df_htri.sort_values("htri", ascending=True),
            x="htri", y="label",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            labels={"htri": "Risk Skoru", "label": "Grup", "risk_level": "Risk"},
            text_auto=".2f",
        )
        fig_bar.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_bar, width="stretch", key="risk_compare_bar")

# ═══════════════════════════════════════════════
# TAB 2 — Koordinasyon Analizi
# ═══════════════════════════════════════════════

with tab2:
    from src.coordination import compute_all_signals, build_interaction_network

    coord_signals = compute_all_signals(df, labels)

    st.subheader("Organizasyon Sinyalleri — Radar Karsilastirmasi")
    st.caption("5 organizasyon sinyalinin anlati bazinda radar karsilastirmasi, sinyal tablosu, anlati etkilesim agi ve hesap yasi dagilimi. Sinyaller: Es Zamanlilik, Hesap Yasi, Metin/Hesap, Cesitlilik, Medya Tekrar.")

    radar_clusters = sorted(
        [s for s in filtered_summaries if s["post_count"] >= 3],
        key=lambda x: x["htri"], reverse=True,
    )[:6]

    fig_radar = go.Figure()
    for s in radar_clusters:
        cid = s["cluster_id"]
        sig = coord_signals.get(cid, {})
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                sig.get("synchrony", 0),
                sig.get("age_clustering", 0),
                sig.get("text_account_ratio", 0),
                1 - sig.get("account_diversity", 1),
                sig.get("media_reuse_score", 0),
            ],
            theta=["Es Zamanlilik", "Hesap Yasi", "Metin / Hesap", "Cesitlilik (ters)", "Medya Tekrar"],
            fill="toself",
            name=s["label"][:25],
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=420, margin=dict(t=20, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_radar, width="stretch", key="coord_radar")

    # ─── Sinyal Tablosu ───
    st.markdown("---")
    st.subheader("Organizasyon Sinyal Degerleri")
    with st.expander("Renkler ne anlama geliyor?"):
        st.markdown(
            "Kirmizi = yuksek risk (0.6+), Turuncu = orta risk (0.3-0.6), Yesil = dusuk risk (0-0.3). "
            "Her sinyal 0-1 arasi deger alir."
        )

    signal_rows = []
    for s in sorted(filtered_summaries, key=lambda x: x["htri"], reverse=True)[:10]:
        if s["post_count"] < 3: continue
        cid = s["cluster_id"]
        sig = coord_signals.get(cid, {})
        signal_rows.append({
            "Grup": s["label"][:30],
            "Risk": _risk_badge(s["risk_level"]),
            "HTRI": f'{s["htri"]:.2f}',
            "Es Zamanlilik": sig.get("synchrony", 0),
            "Hesap Yasi": sig.get("age_clustering", 0),
            "Metin/Hesap": sig.get("text_account_ratio", 0),
            "Cesitlilik (1-x)": 1 - sig.get("account_diversity", 1),
            "Medya Tekrar": sig.get("media_reuse_score", 0),
        })

    if signal_rows:
        sig_df = pd.DataFrame(signal_rows)
        for col in ["Es Zamanlilik", "Hesap Yasi", "Metin/Hesap", "Cesitlilik (1-x)", "Medya Tekrar"]:
            sig_df[col] = sig_df[col].apply(lambda v: _signal_bar(v))
        st.markdown(sig_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ─── Network Graph ───
    st.markdown("---")
    st.subheader("Anlati Etkilesim Agi")
    st.caption("Dugumler = anlati gruplari. Kenarlar = ortak hesap paylasimi (kalinlik = paylasilan hesap sayisi). Kirmizi = kritik risk.")

    G = build_interaction_network(df, labels, min_edge_weight=1)
    if G.number_of_nodes() > 0:
        lookup = {s["cluster_id"]: s for s in filtered_summaries}
        pos = nx.spring_layout(G, seed=42, k=0.4)

        edge_trace = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            weight = d.get("weight", 1)
            edge_trace.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=min(weight * 0.8, 6), color=f"rgba(150,100,150,{min(weight*0.08, 0.6)})"),
                hoverinfo="text", text=f"ortak hesap: {weight}",
            ))

        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            s = lookup.get(node, {})
            node_text.append(s.get("label", f"Grup_{node}")[:25])
            sz = s.get("post_count", 10) / 3
            node_size.append(max(sz, 10))
            node_color.append(RISK_COLORS.get(s.get("risk_level", ""), "#95A5A6"))

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=node_text, textposition="top center",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
            hoverinfo="text",
        )
        fig_net = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title="Anlati Etkilesim Agi (kenar = ortak hesap sayisi)",
                showlegend=False, hovermode="closest",
                margin=dict(b=10, l=10, r=10, t=40),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=500,
            ),
        )
        st.plotly_chart(fig_net, width="stretch", key="coord_network")
    else:
        st.info("Etkilesim agi olusturmak icin yeterli kume bulunamadi.")

    # ─── Account-Based Network Graph ───
    st.markdown("---")
    st.subheader("Hesap Bazli Etkilesim Agi")
    st.caption("Secili kumedeki hesaplar. Dugum = hesap, kenar = ortak medya kodu, buyukluk = post sayisi, renk = hesap yasi (koyu = eski).")

    acct_network_options = [s for s in filtered_summaries if s["post_count"] >= 3]
    if acct_network_options:
        selected_label = st.selectbox(
            "Kume Sec",
            options=[s["cluster_id"] for s in acct_network_options],
            format_func=lambda cid: next(
                (s["label"][:40] for s in acct_network_options if s["cluster_id"] == cid),
                f"Grup_{cid}",
            ),
            key="acct_net_select",
        )

        selected_summary = next(s for s in acct_network_options if s["cluster_id"] == selected_label)
        cdf = df[df["cluster_id"] == selected_label]
        user_posts = cdf.groupby("user_id").agg(
            post_count=("text", "count"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
            account_age=("account_created_at", "first"),
        ).reset_index()

        G_acct = nx.Graph()
        for _, row in user_posts.iterrows():
            G_acct.add_node(
                row["user_id"],
                size=row["post_count"],
                age=row["account_age"],
                label=row["user_id"][:10],
            )

        if "media_hash" in cdf.columns:
            media_by_user = cdf.groupby("user_id")["media_hash"].apply(
                lambda x: set(x.dropna().unique())
            ).to_dict()
            users = list(media_by_user.keys())
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    shared = media_by_user[users[i]] & media_by_user[users[j]]
                    if shared:
                        G_acct.add_edge(users[i], users[j], weight=len(shared))

        if G_acct.number_of_nodes() > 0:
            pos = nx.spring_layout(G_acct, seed=42, k=0.6)
            node_x, node_y, node_sizes, node_colors, node_texts = [], [], [], [], []
            age_min = user_posts["account_age"].min()
            age_max = user_posts["account_age"].max()
            age_range = max((age_max - age_min).days, 1)

            for node in G_acct.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                nd = G_acct.nodes[node]
                sz = nd.get("size", 1) * 12
                node_sizes.append(max(sz, 15))
                age_days = (nd.get("age", age_min) - age_min).days
                age_ratio = age_days / age_range
                node_colors.append(age_ratio)
                node_texts.append(f"Hesap: {nd.get('label', node)}<br>Post: {nd.get('size', 0)}")

            edge_traces_acct = []
            for u, v, d in G_acct.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                w = d.get("weight", 1)
                edge_traces_acct.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=min(w * 2, 6), color="rgba(150,100,150,0.3)"),
                    hoverinfo="text",
                    text=f"ortak medya: {w}",
                ))

            node_trace_acct = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                text=node_texts,
                hoverinfo="text",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale="RdYlBu_r",
                    colorbar=dict(title="Hesap Yasiligi", tickvals=[0, 1], ticktext=["Yeni", "Eski"]),
                    line=dict(width=1, color="white"),
                    showscale=True,
                ),
            )

            fig_acct = go.Figure(
                data=edge_traces_acct + [node_trace_acct],
                layout=go.Layout(
                    title=f"Hesap Agi: {selected_summary['label'][:40]}",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=10, l=10, r=10, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, visible=False),
                    yaxis=dict(showgrid=False, zeroline=False, visible=False),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=500,
                ),
            )
            st.plotly_chart(fig_acct, width="stretch", key="acct_network")
        else:
            st.info("Bu kume icin hesap agi olusturulamadi.")

    # ─── Account Age Distribution ───
    st.markdown("---")
    st.subheader("Hesap Yasi Dagilimi")
    acct_info = (
        df_visible.groupby(["user_id", "cluster_id"])["account_created_at"]
        .first().reset_index()
    )
    fig_age = px.histogram(
        acct_info, x="account_created_at",
        color="cluster_id",
        nbins=30,
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={"account_created_at": "Hesap Olusturma Tarihi", "count": "Hesap Sayisi", "cluster_id": "Grup"},
    )
    fig_age.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_age, width="stretch", key="account_age_hist")

# ═══════════════════════════════════════════════
# TAB 3 — Dogrulama Sinyalleri
# ═══════════════════════════════════════════════

with tab3:
    st.subheader("Medya Tekrar Kullanimi Analizi")
    st.caption("Gorsel benzerlik karsilastirmasi ile medya tekrari tespiti. Koordineli dezenformasyonda ayni gorsel farkli hesaplarca tekrar kullanilir.")

    from src.media_reuse import compute_media_reuse, build_media_network

    mr_scores = compute_media_reuse(df, labels)
    mr_edges = build_media_network(df, labels)

    chart_data = [s for s in filtered_summaries if s["post_count"] >= 3]

    # --- Media Reuse Score bar chart ---
    if chart_data:
        reuse_rows = []
        for s in chart_data:
            cid = s["cluster_id"]
            m = mr_scores.get(cid, {})
            reuse_rows.append({
                "Grup": s["label"][:30],
                "Grup Ici Ortak Medya": m.get("intra_overlap", 0),
                "Gruplar Arasi Ortak Medya": m.get("cross_overlap", 0),
                "Medya Tekrar Skoru": m.get("media_reuse_score", 0),
                "Risk": s["risk_level"],
            })
        if reuse_rows:
            df_mr = pd.DataFrame(reuse_rows)
            fig_mr = px.bar(
                df_mr.sort_values("Medya Tekrar Skoru"),
                x="Medya Tekrar Skoru", y="Grup",
                color="Risk", color_discrete_map=RISK_COLORS,
                text_auto=".2f",
                labels={"Medya Tekrar Skoru": "Medya Tekrar Skoru"},
            )
            fig_mr.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_mr, width="stretch", key="media_reuse_bar")

            st.markdown("---")

            # --- Intra/Cross overlap grouped bar ---
            df_melt = df_mr.melt(
                id_vars=["Grup"],
                value_vars=["Grup Ici Ortak Medya", "Gruplar Arasi Ortak Medya"],
                var_name="Tur", value_name="Oran",
            )
            fig_overlap = px.bar(
                df_melt, x="Oran", y="Grup", color="Tur",
                barmode="group", text_auto=".2f",
                color_discrete_map={
                    "Grup Ici Ortak Medya": "#8E44AD",
                    "Gruplar Arasi Ortak Medya": "#D55E00",
                },
            )
            fig_overlap.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_overlap, width="stretch", key="media_overlap_bar")

    st.markdown("---")

    # --- Media Sharing Network ---
    st.subheader("Medya Kod Paylasim Agi")
    st.caption("Dugumler = anlati kumeleri. Kenarlar = ortak medya kodu (kalinlik = kac farkli medya). Farkli anlatilar ayni gorseli kullaniyorsa bu guclu koordinasyon isaretidir.")

    if mr_edges:
        G_mr = nx.Graph()
        lookup = {s["cluster_id"]: s for s in chart_data}
        all_nodes = set(u for u, v in mr_edges) | set(v for u, v in mr_edges)
        for cid in all_nodes:
            s = lookup.get(cid, {})
            G_mr.add_node(
                cid,
                label=s.get("label", f"Grup_{cid}")[:25],
                color=RISK_COLORS.get(s.get("risk_level", ""), "#95A5A6"),
                size=s.get("post_count", 10),
            )
        for (u, v), w in mr_edges.items():
            G_mr.add_edge(u, v, weight=w)

        pos = nx.spring_layout(G_mr, seed=42, k=0.5)
        edge_traces_mr = []
        for u, v, d in G_mr.edges(data=True):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_traces_mr.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=min(d.get("weight", 1) * 3, 8), color="rgba(142,68,173,0.3)"),
                hoverinfo="text", text=f"ortak medya: {d.get('weight', 1)}",
            ))

        node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
        for node in G_mr.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            nd = G_mr.nodes[node]
            node_text.append(nd.get("label", f"Grup_{node}"))
            sz = nd.get("size", 10) / 3
            node_size.append(max(sz, 10))
            node_color.append(nd.get("color", "#95A5A6"))

        node_trace_mr = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            text=node_text, textposition="top center",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
            hoverinfo="text",
        )
        fig_mr_net = go.Figure(
            data=edge_traces_mr + [node_trace_mr],
            layout=go.Layout(
                title="Medya Kod Paylasimi (kenar = ortak medya kodu sayisi)",
                showlegend=False, hovermode="closest",
                margin=dict(b=10, l=10, r=10, t=40),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                height=450,
            ),
        )
        st.plotly_chart(fig_mr_net, width="stretch", key="media_network")
    else:
        st.info("Medya paylasim agi icin yeterli veri bulunamadi.")

    st.markdown("---")
    st.subheader("Risk Skoru Bilesenleri")

    if chart_data:
        df_htri2 = pd.DataFrame(chart_data)
        fig_htri = px.bar(
            df_htri2.sort_values("htri"),
            x="htri", y="label", color="risk_level",
            color_discrete_map=RISK_COLORS,
            labels={"htri": "Risk Skoru", "label": "Grup", "risk_level": "Risk"},
            text_auto=".2f",
        )
        fig_htri.update_layout(height=350, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_htri, width="stretch", key="htri_components")

    with st.expander("Risk Skoru (HTRI) Hesaplama Formulu"):
        st.markdown("""
**Hybrid Threat Risk Index (HTRI)** — Cifte sayim yok, her bilesen bagimsiz

```
HTRI = 0.30 x Koordinasyon Skoru
     + 0.25 x Medya Tekrar Skoru
     + 0.25 x Ivme (Acceleration)
     + 0.20 x Hesap Yasi Riski
```

*Ivme: posting hizinin zaman icindeki ivmesi (pozitif = patlama buyumesi). Hesap Yasi Riski sadece HTRI'de yer alir, Coordination'da degil.*

**Koordinasyon Skoru (3 davranissal sinyal):**
```
Coord = 0.40 x Es Zamanlilik
      + 0.35 x Metin Benzerligi / Hesap Orani
      + 0.25 x (1 - Hesap Cesitliligi)
```

*Medya Tekrari ve Hesap Yasi koordinasyon skorundan cikarilmistir, HTRI'de bagimsiz bilesen olarak toplanir.*

**Medya Tekrar Kullanim Skoru:**
```
MediaReuse = 0.60 x Kume Ici Ortaklik + 0.40 x Kumeler Arasi Ortaklik
```

**Risk Seviyeleri:**
- 0.00-0.30 -> Dusuk
- 0.30-0.60 -> Orta
- 0.60-0.80 -> Yuksek
- 0.80-1.00 -> Kritik
""")

    st.markdown("---")
    st.subheader("Veri Seti Bilgisi")
    n_clusters = len([s for s in summaries if not s["is_noise"]])
    n_noise = sum(1 for s in summaries if s["is_noise"])

    st.markdown(f"""
- **Toplam kayit:** {len(df)} (filtrelenmis: {len(df_visible)})
- **Benzersiz hesap:** {all_accounts}
- **Grup sayisi:** {n_clusters} (+ {n_noise} siniflandirilmayan)
- **Kaynak:** Sentetik veri
""")

st.markdown("---")
st.caption(
    "FIMI-TR v0.5 | NLP: paraphrase-multilingual-MiniLM-L12-v2 | Gruplama: DBSCAN | Medya: Gorsel Benzerlik | Tamamen acik kaynak"
)
st.markdown(
    '<div style="font-size:11px;color:#999;text-align:center;padding:10px 0 4px;border-top:1px solid #eee;margin-top:4px;">'
    "⚠\uFE0F Bu bir erken uyar\u0131 sistemidir, kesin do\u011fruluk garantisi vermez. "
    "Nihai karar\u0131n\u0131z\u0131 resmi kaynaklardan teyit ederek veriniz."
    "</div>",
    unsafe_allow_html=True,
)
