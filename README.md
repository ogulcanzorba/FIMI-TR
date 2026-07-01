# FIMI-TR — Türkçe Bilgi Manipülasyonu Erken Uyarı Platformu
### Foreign Information Manipulation and Interference — Early Warning Platform for Turkish Content

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface" />
  <img src="https://img.shields.io/badge/Lisans-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Hackathon-TEKNOPARK%2024h-orange?style=flat-square" />
</p>

---

## 🇹🇷 Türkçe

### Nedir?

FIMI-TR, Türkçe sosyal medya ve haber içeriklerindeki koordineli dezenformasyon anlatılarını tespit eden, risk skorlayan ve zaman içindeki evrimini izleyen açık kaynaklı bir erken uyarı platformudur.

Proje, Avrupa Birliği Dış İlişkiler Servisi'nin (EEAS) tanımladığı **FIMI (Foreign Information Manipulation and Interference)** çerçevesinden ilham almakta; bu tehdide Türkçe içerik için ölçeklenebilir, açık kaynaklı bir cevap sunmayı hedeflemektedir.

### Neden Gerekli?

Koordineli dezenformasyon kampanyaları; kriz anlarında (deprem, seçim, ekonomik şok) ani yayılım gösterir, gerçek haberin önüne geçer ve toplumsal algıyı manipüle eder. Akademik çalışmalar bu tehdidi belgelemektedir — ancak Türkçe içerik için canlı, kullanıcıya açık, açık kaynaklı bir tespit aracı mevcut değildir.

> *"FIMI'nin sosyal medya ölçeğinde tespiti hâlâ büyük ölçüde manuel ve ölçeklenmesi zor bir iştir."*
> — arXiv, 2026

### Nasıl Çalışır?

```
İçerik Girişi (metin / haber başlığı)
        ↓
Çok Dilli Embedding (sentence-transformers)
        ↓
Anlatı Kümeleme (DBSCAN, cosine mesafesi)
        ↓
Koordinasyon Analizi (5 davranışsal sinyal)
        ↓
HTRI — Hibrit Tehdit Risk İndeksi (0.00 – 1.00)
        ↓
Erken Uyarı Dashboard (Streamlit)
```

### Risk Skoru: HTRI Formülü

```
HTRI = 0.30 × Koordinasyon Skoru
     + 0.25 × Medya Tekrar Skoru
     + 0.25 × İvme (Acceleration)
     + 0.20 × Hesap Yaşı Riski
```

**Koordinasyon Skoru:**
```
Coord = 0.40 × Eş Zamanlılık
      + 0.35 × Metin/Hesap Oranı
      + 0.25 × (1 − Hesap Çeşitliliği)
```

| HTRI Aralığı | Risk Seviyesi |
|---|---|
| 0.00 – 0.30 | 🟢 Düşük |
| 0.30 – 0.60 | 🟡 Orta |
| 0.60 – 0.80 | 🟠 Yüksek |
| 0.80 – 1.00 | 🔴 Kritik |

### Teknoloji Yığını

| Bileşen | Teknoloji | Neden? |
|---|---|---|
| Arayüz | Streamlit + Plotly | Hızlı prototipleme, native grafik desteği |
| NLP | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` | Türkçe dahil 50+ dil, CPU'da hızlı |
| Kümeleme | scikit-learn DBSCAN (cosine, eps=0.45) | Küme sayısı otomatik, gürültüye dayanıklı |
| Görselleştirme | t-SNE (2D projeksiyon) | Anlatı benzerliğini görsel olarak sunar |
| Ağ Analizi | NetworkX | Hesap etkileşim ağı, koordinasyon grafiği |
| Medya Tespiti | imagehash (perceptual hash) | Görsel tekrar kullanımını skorlar |
| Etiketleme | TF-IDF (top-3 terim) | Açıklanabilir, deterministik |
| Veri | Sentetik + gerçek Türkçe tweet + haber | Ground-truth ile doğrulanmış |

> Tüm bileşenler **ücretsiz ve açık kaynaklıdır.** Hiçbir ücretli API veya servis kullanılmamaktadır.

### Kurulum

```bash
git clone https://github.com/KULLANICI_ADIN/FIMI-TR.git
cd FIMI-TR/project
pip install -r requirements.txt
streamlit run app.py
```

İlk çalıştırmada NLP modeli (~500 MB) HuggingFace'ten otomatik indirilir, sonrasında önbellekten yüklenir.

### Doğrulama

Pipeline, iki farklı gerçek veri kümesiyle test edilmiştir:

| Veri Kümesi | Kayıt | Sonuç |
|---|---|---|
| FatmaElik/turkish-disaster-news-geonlp | 472 haber | Küme tutarlılığı: hasar %83, aciliyet %94 |
| yankihue/tweets-turkish | 11.108 tweet | Organik küme karşılaştırması |
| Sentetik (ground-truth) | 433 gönderi | Kampanya HTRI=0.77 vs Organik HTRI=0.21 |

### Proje Yapısı

```
project/
├── app.py                     # Ana Streamlit dashboard (4 sekme)
├── requirements.txt
├── data/
│   ├── synthetic_data.csv     # Sentetik + gerçek tweet karışımı
│   ├── real_sample.csv        # FatmaElik haber verisi
│   ├── real_data_validation.md
│   └── embeddings/            # Embedding önbelleği (.npy)
└── src/
    ├── nlp_pipeline.py        # Embedding, DBSCAN, t-SNE, TF-IDF
    ├── coordination.py        # 5 koordinasyon sinyali + ağ
    ├── htri_timeline.py       # Kümülatif HTRI zaman serisi
    ├── media_reuse.py         # Perceptual hash benzerliği
    └── narrative_evolution.py # Anlatı evrimi analizi
```

### Bilinen Sınırlamalar

- Mevcut veri kümesi sentetik ağırlıklıdır; gerçek zamanlı sosyal medya bağlantısı henüz entegre edilmemiştir.
- Bu bir **erken uyarı sistemidir** — kesin doğruluk garantisi vermez, insan analistinin kararını desteklemek için tasarlanmıştır.
- LLM tabanlı etiketleme altyapısı mevcut (`src/llm_labeler.py`) ancak CPU performans kısıtı nedeniyle varsayılan olarak devre dışıdır.

---

## 🇬🇧 English

### What is it?

FIMI-TR is an open-source early warning platform that detects, scores, and tracks coordinated disinformation narratives in Turkish-language social media and news content.

The project is inspired by the **FIMI (Foreign Information Manipulation and Interference)** framework defined by the European External Action Service (EEAS), and aims to provide a scalable, open-source response to this threat for Turkish-language content — a gap currently unaddressed by existing tools.

### Why is it needed?

Coordinated disinformation campaigns tend to surge during crises (earthquakes, elections, economic shocks), outrunning factual reporting and manipulating public perception. While academic work documents this threat extensively, no live, publicly accessible, open-source detection tool exists for Turkish-language content.

> *"Operationalizing FIMI detection at social media scale remains largely manual and interpretation-heavy."*
> — arXiv, 2026

### How it works

```
Content Input (text / news headline)
        ↓
Multilingual Embedding (sentence-transformers)
        ↓
Narrative Clustering (DBSCAN, cosine distance)
        ↓
Coordination Analysis (5 behavioral signals)
        ↓
HTRI — Hybrid Threat Risk Index (0.00 – 1.00)
        ↓
Early Warning Dashboard (Streamlit)
```

### Risk Score: HTRI Formula

```
HTRI = 0.30 × Coordination Score
     + 0.25 × Media Reuse Score
     + 0.25 × Acceleration
     + 0.20 × Account Age Risk
```

**Coordination Score:**
```
Coord = 0.40 × Synchrony
      + 0.35 × Text/Account Ratio
      + 0.25 × (1 − Account Diversity)
```

| HTRI Range | Risk Level |
|---|---|
| 0.00 – 0.30 | 🟢 Low |
| 0.30 – 0.60 | 🟡 Medium |
| 0.60 – 0.80 | 🟠 High |
| 0.80 – 1.00 | 🔴 Critical |

### Tech Stack

| Component | Technology | Why |
|---|---|---|
| UI | Streamlit + Plotly | Rapid prototyping, native chart support |
| NLP | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` | 50+ languages including Turkish, fast on CPU |
| Clustering | scikit-learn DBSCAN (cosine, eps=0.45) | Auto cluster count, noise-robust |
| Visualization | t-SNE (2D projection) | Visual narrative similarity map |
| Network | NetworkX | Account interaction graph |
| Media | imagehash (perceptual hashing) | Visual reuse scoring |
| Labeling | TF-IDF (top-3 terms) | Explainable, deterministic |
| Data | Synthetic + real Turkish tweets + news | Validated against ground-truth |

> All components are **free and open-source.** No paid APIs or services are used.

### Installation

```bash
git clone https://github.com/KULLANICI_ADIN/FIMI-TR.git
cd FIMI-TR/project
pip install -r requirements.txt
streamlit run app.py
```

The NLP model (~500 MB) is automatically downloaded from HuggingFace on first run and cached locally thereafter.

### Validation

The pipeline was tested against two real-world Turkish datasets:

| Dataset | Records | Result |
|---|---|---|
| FatmaElik/turkish-disaster-news-geonlp | 472 news articles | Cluster purity: damage 83%, urgency 94% |
| yankihue/tweets-turkish | 11,108 tweets | Organic cluster baseline |
| Synthetic (ground-truth) | 433 posts | Campaign HTRI=0.77 vs Organic HTRI=0.21 |

### Known Limitations

- The current dataset is primarily synthetic; real-time social media ingestion is not yet integrated.
- This is an **early warning system** — it does not guarantee accuracy and is designed to support human analyst decisions, not replace them.
- LLM-based labeling infrastructure exists (`src/llm_labeler.py`) but is disabled by default due to CPU latency constraints.

---

## Hackathon Context

Bu proje, **TEKNOPARK İstanbul** bünyesinde düzenlenen **Hibrit Tehditler** temalı 24 saatlik hackathon kapsamında, tek kişi tarafından geliştirilmiştir.

*This project was developed solo during a 24-hour hackathon on the theme of Hybrid Threats, organized by TEKNOPARK İstanbul.*

---

## Kaynaklar / References

- [EU Council — Hybrid Threats](https://www.consilium.europa.eu/en/policies/hybrid-threats)
- [EEAS — FIMI Framework](https://www.eeas.europa.eu/eeas/information-integrity-and-countering-foreign-information-manipulation-interference-fimi_en)
- [MİT Sözlük — OSINT Tanımı](https://mit.gov.tr/sozluk.html)
- [FatmaElik/turkish-disaster-news-geonlp](https://huggingface.co/datasets/FatmaElik/turkish-disaster-news-geonlp)
- arXiv, 2026 — *An Agentic Operationalization of DISARM for FIMI Investigation on Social Media*

---

<p align="center">
Tamamen ücretsiz ve açık kaynaklı araçlarla geliştirilmiştir.<br>
<em>Built entirely with free and open-source tools.</em>
</p>
