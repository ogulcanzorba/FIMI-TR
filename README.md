# FIMI-TR

Türkçe haber ve sosyal medya içeriklerindeki koordineli dezenformasyon girişimlerini tespit edip erken uyarı sağlayan açık kaynak bir platform.

## Kurulum

```bash
cd project
pip install -r requirements.txt
streamlit run app.py
```

## Teknoloji Yığını

| Bileşen | Teknoloji |
|---------|-----------|
| UI | Streamlit + Plotly + NetworkX |
| NLP | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| Kümeleme | DBSCAN (cosine, eps=0.45, min_samples=2) |
| Görselleştirme | t-SNE (2D projeksiyon) |
| Risk Skoru (HTRI) | Coord(0.30) + MediaReuse(0.25) + Acceleration(0.25) + AgeRisk(0.20) |
| Coordination | Synchrony(0.40) + TextAccountRatio(0.35) + (1-Diversity)(0.25) |
| Medya | Perceptual hash (intra + cross overlap) |
| Etiketleme | TF-IDF (top-3 terim) |
| Veri | Sentetik + gerçek Türkçe tweet'ler |

Tamamen ücretsiz ve açık kaynak araçlarla geliştirilmiştir.

---

*Hibrit Tehditler temalı 24 saatlik TEKNOPARK hackathonu kapsamında geliştirilmiştir.*
