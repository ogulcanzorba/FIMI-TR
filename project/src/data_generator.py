import csv
import random
import hashlib
import os
import pickle
import sys
from datetime import datetime, timedelta

random.seed(42)

_REAL_TEXTS_CACHE = None


def _load_real_texts() -> list[str]:
    global _REAL_TEXTS_CACHE
    if _REAL_TEXTS_CACHE is not None:
        return _REAL_TEXTS_CACHE
    pkl_path = os.path.join(os.path.dirname(__file__), "..", "data", "real_tweet_texts.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            _REAL_TEXTS_CACHE = pickle.load(f)
        print(f"[data_generator] Gercek tweet metinleri yuklendi: {len(_REAL_TEXTS_CACHE)} adet")
    else:
        _REAL_TEXTS_CACHE = []
    return _REAL_TEXTS_CACHE


def _real_texts_for_organic(n: int) -> list[str]:
    all_texts = _load_real_texts()
    if not all_texts:
        return []
    return random.sample(all_texts, min(n, len(all_texts)))


# Fallback dummy texts (eger real_tweet_texts.pkl yoksa kullanilir)
CAMPAIGN_1_TEXTS = [
    "Elektrik kesintisi tum sehirde devam ediyor, yetkililerden aciklama yok",
    "Saatlerdir elektrik yok, kimse nedenini soylemiyor",
    "#ElektrikKesintisi sehri karanliga bogdu, hala aciklama yapilmadi",
    "Elektrik kesintisiyle ilgili resmi bir aciklama gelmedi",
    "Yetkililer elektrik kesintisi hakkinda sessizligini koruyor",
    "Sehir genelinde elektrik yok, panik buyuyor",
    "Elektrik kesintisi 8 saati asti, kimse ne oldugunu bilmiyor",
    "Hastaneler jenatorle calisiyor, elektrik hala gelmedi",
    "Elektrik kesintisi kriz yonetimi tamamen basarisiz",
    "Ulusal sebeke cokuyor, hukumet krizi gizliyor",
    "Sebeke coktu, hukumet aciklama yapmaktan kaciniyor",
    "Elektrik dagitim sirketleri krizi yonetimeiyor",
    "Kesintinin nedeni siber saldiri olabilir, yetkililer dogrulamiyor",
    "Milyonlarca insan elektriksiz, kimse sorumluluk almiyor",
    "Bu buyuklukte bir kesinti daha once yasanmamisti",
    "Elektrik kesintisi ekonomik kayiplara yol aciyor",
    "Karaborsacilar jenator fiyatlarini 5 katina cikardi",
    "Okullar elektrik yokken tatil edildi, resmi aciklama bekleniyor",
    "Sular da kesildi, cunku pompalar calismiyor",
    "Elektrik gelmeyince hayat durma noktasina geldi",
]

CAMPAIGN_2_TEXTS = [
    "Deprem olacagi kesin! Uzmanlar uyariyor: Buyuk Marmara depremi cok yakin",
    "AFET koordinasyon merkezi gizli toplanti yapti, deprem hazirligi basladi",
    "ABD'li uzmanlar 7.4 buyuklugunde deprem uyarisi yapti",
    "Deprem dedektorleri olagan aktivite tespit etti",
    "Fay hattinda hareketlilik var, bu hafta icinde deprem bekleniyor",
    "Deprem hazirliklari icin kriz masasi kuruldu",
    "Japonya'dan gelen deprem uzmanlari Istanbul'da incelemelere basladi",
    "Deprem sigortasi yaptirmayanlar pisman olacak",
    "Bilim insanlari susuyor ama deprem kacinilmaz",
    "Prof. Dr. X: 48 saat icinde buyuk deprem olabilir",
]

ORGANIC_TEXTS = [
    "Bugun hava cok guzeldi, sahilde yuruyus yaptim",
    "Yeni tarif denedim, makarna harika oldu",
    "Bugun ise giderken trafik yogundu",
    "Kedim hastalandi, veterinere goturmek lazim",
    "Bu kitabi kesinlikle okumalisiniz, harika",
    "Dizi onerisi olan var mi? Yeni ne izlesem bilemedim",
    "Sabah sporu yapmak insana iyi geliyor",
    "Bugun yagmur yagdi, semsiyemi unutmusum",
    "Yeni bir kafe kesfettim, kahveleri cok iyi",
    "Hafta sonu plani yapana kadar aksam oldu",
    "Cocukken oynadigim oyunlari hatirladim da ne guzel gunlerdi",
    "Bugun mac vardi, bizim takim kazandi",
    "Pazardan meyve aldim, cok taze",
    "Yogun bir gundu, sonunda eve geldim",
    "Muzik dinlemek her seye iyi geliyor",
    "Bu aralar cok yogunum, yetismeye calisiyorum",
    "Istanbul'da yeni bir sergi acildi, gormeye gidecegim",
    "Teknoloji harika seyler, yeni telefonum geldi",
    "Komsu davetiye gondermis, aksam yemege gidecegiz",
    "Sinemaya gittim film cok iyiydi",
    "Yilbasi hazirliklari basladi, evi susleyecegim",
    "Sabah erken kalkmak zor ama aliskanlik oluyor",
    "Arkadasimla uzun zamandir gorusemiyorduk bugun bulustuk",
    "Evdeki bitkiler buyudu, saksı degistirme vakti",
    "Deniz kenarinda yurumek insanin icini aciyor",
    "Bu hafta sonu kamp yapmaya gidiyoruz heyecanliyim",
    "Yeni bir hobi edindim, resim yapmaya basladim",
    "Cayin yanina ne yapsam diye dusunuyordum, kek yaptim",
    "Aksam yuruyusu yapmak uyku kalitesini artiriyor",
    "Bugun cok mutluyum, guzel haber aldim",
]

PLATFORMS = ["Twitter", "Telegram", "Haber"]


def generate_user_id(prefix: str, index: int) -> str:
    raw = f"{prefix}{index:04d}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def pick_text(texts, used_texts, similarity=0.3):
    if random.random() < similarity and used_texts:
        return random.choice(used_texts)
    t = random.choice(texts)
    used_texts.append(t)
    return t


def _media_hash_campaign(campaign_id: str, pool_size: int) -> str:
    return hashlib.md5(f"{campaign_id}_img{random.randint(1, pool_size)}".encode()).hexdigest()[:16]


def _campaign_text_pool(prefix: str = "camp", count: int = 25) -> list[str]:
    """Use real texts if available, else fallback to hardcoded campaign texts."""
    all_real = _load_real_texts()
    if all_real:
        seed = sum(ord(c) for c in prefix)
        rng = random.Random(seed)
        pool = rng.sample(all_real, min(count, len(all_real)))
        print(f"  [campaign {prefix}] using {len(pool)} real tweet texts")
        return pool
    return []


def generate_campaign_1(base_time: datetime) -> list[dict]:
    """Coordinated Campaign 1: same narrative variations, same-day accounts, tight timestamps"""
    rows = []
    account_creation = base_time - timedelta(days=random.randint(0, 2))
    accounts = [generate_user_id("haber", i) for i in range(8)]
    used_texts = []
    post_id_counter = 1

    # Kampanyalar orijinal metinlerle kaliyor (konu butunlugu icin)
    pool1 = CAMPAIGN_1_TEXTS

    hours = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    for h in hours:
        for _ in range(random.randint(2, 5)):
            ts = base_time.replace(hour=h, minute=random.randint(0, 59))
            user = random.choice(accounts)
            text = pick_text(pool1, used_texts, similarity=0.6)
            platform = random.choices(PLATFORMS, weights=[0.55, 0.30, 0.15], k=1)[0]
            mentions = []
            if random.random() < 0.3:
                mentions.append(random.choice(accounts))
            repost_of = None
            if random.random() < 0.25:
                repost_of = f"P-{post_id_counter - random.randint(1, 5):04d}"
            media_hash = _media_hash_campaign("camp1", 10)
            rows.append({
                "post_id": f"P-{post_id_counter:04d}",
                "user_id": user,
                "text": text,
                "timestamp": ts.isoformat(),
                "account_created_at": account_creation.isoformat(),
                "source_platform": platform,
                "mentions": ",".join(mentions) if mentions else "",
                "repost_of": repost_of or "",
                "media_hash": media_hash,
                "cluster": "Kampanya1",
            })
            post_id_counter += 1
    return rows


def generate_campaign_2(base_time: datetime) -> list[dict]:
    """Coordinated Campaign 2: few accounts, high volume, high text similarity"""
    rows = []
    accounts = [generate_user_id("deprem", i) for i in range(5)]
    account_creation = base_time - timedelta(days=random.randint(0, 3))
    used_texts = []
    post_id_counter = 1

    pool2 = CAMPAIGN_2_TEXTS

    hours = list(range(6, 23))
    for h in hours:
        for _ in range(random.randint(3, 8)):
            ts = base_time.replace(hour=h, minute=random.randint(0, 59))
            user = random.choice(accounts)
            text = pick_text(pool2, used_texts, similarity=0.8)
            platform = random.choices(PLATFORMS, weights=[0.40, 0.50, 0.10], k=1)[0]
            mentions = []
            if random.random() < 0.5:
                mentions.append(random.choice(accounts))
            repost_of = None
            if random.random() < 0.4:
                repost_of = f"P-{post_id_counter - random.randint(1, 8):04d}"
            media_hash = _media_hash_campaign("camp2", 5)
            rows.append({
                "post_id": f"P-{post_id_counter:04d}",
                "user_id": user,
                "text": text,
                "timestamp": ts.isoformat(),
                "account_created_at": account_creation.isoformat(),
                "source_platform": platform,
                "mentions": ",".join(mentions) if mentions else "",
                "repost_of": repost_of or "",
                "media_hash": media_hash,
                "cluster": "Kampanya2",
            })
            post_id_counter += 1
    return rows


def generate_organic(base_time: datetime) -> list[dict]:
    """Organic cluster: many users, low similarity, spread timestamps.
    Her kullanici gunde en fazla 1 post atar — zaman kumelenmesini onler."""
    rows = []
    accounts = []
    account_creation_base = base_time - timedelta(days=180)

    for i in range(60):
        uid = generate_user_id("kullanici", i)
        created = account_creation_base + timedelta(days=random.randint(0, 150))
        accounts.append((uid, created))

    used_texts = []
    post_id_counter = 1

    real_organic = _real_texts_for_organic(500)
    pool_organic = real_organic if real_organic else ORGANIC_TEXTS
    if real_organic:
        print(f"  [organic] using {len(real_organic)} real tweet texts in rotation")

    for day_offset in range(30):
        day_time = base_time - timedelta(days=29 - day_offset)
        num_posts = random.randint(5, 15)
        chosen = random.sample(accounts, min(num_posts, len(accounts)))
        for user, created in chosen:
            ts = day_time.replace(hour=random.randint(7, 23), minute=random.randint(0, 59))
            text = pick_text(pool_organic, used_texts, similarity=0.05)
            platform = random.choices(PLATFORMS, weights=[0.60, 0.15, 0.25], k=1)[0]
            mentions = []
            if random.random() < 0.15:
                mentions.append(random.choice([a[0] for a in accounts]))
            repost_of = None
            if random.random() < 0.1:
                repost_of = f"P-{post_id_counter - random.randint(1, 15):04d}"
            if random.random() < 0.05:
                media_hash = hashlib.md5(f"organic_img{random.randint(1, 3)}".encode()).hexdigest()[:16]
            else:
                media_hash = hashlib.md5(f"organic_{post_id_counter}".encode()).hexdigest()[:16]
            rows.append({
                "post_id": f"P-{post_id_counter:04d}",
                "user_id": user,
                "text": text,
                "timestamp": ts.isoformat(),
                "account_created_at": created.isoformat(),
                "source_platform": platform,
                "mentions": ",".join(mentions) if mentions else "",
                "repost_of": repost_of or "",
                "media_hash": media_hash,
                "cluster": "Organik",
            })
            post_id_counter += 1
    return rows


def generate_dataset(output_path: str = "data/synthetic_data.csv"):
    now = datetime.now()

    print("[1/3] Kampanya 1 (Koordine) olusturuluyor...")
    camp1 = generate_campaign_1(now - timedelta(days=1))

    print("[2/3] Kampanya 2 (Koordine) olusturuluyor...")
    camp2 = generate_campaign_2(now - timedelta(days=2))

    print("[3/3] Organik kume olusturuluyor...")
    organic = generate_organic(now)

    all_rows = camp1 + camp2 + organic
    random.shuffle(all_rows)

    fieldnames = [
        "post_id", "user_id", "text", "timestamp",
        "account_created_at", "source_platform",
        "mentions", "repost_of", "media_hash", "cluster",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nVeri seti olusturuldu: {output_path}")
    print(f"Toplam kayit: {len(all_rows)}")
    print(f"  Kampanya 1 (Koordine): {len(camp1)}")
    print(f"  Kampanya 2 (Koordine): {len(camp2)}")
    print(f"  Organik kume:          {len(organic)}")

    clusters = {}
    for r in all_rows:
        c = r["cluster"]
        clusters[c] = clusters.get(c, 0) + 1
    for c, cnt in sorted(clusters.items()):
        unique_accounts = set(r["user_id"] for r in all_rows if r["cluster"] == c)
        platforms = {}
        for r in all_rows:
            if r["cluster"] == c:
                p = r["source_platform"]
                platforms[p] = platforms.get(p, 0) + 1
        print(f"\n  [{c}]")
        print(f"    Gonderen: {cnt}")
        print(f"    Benzersiz hesap: {len(unique_accounts)}")
        print(f"    Platform dagilimi: {platforms}")

    return all_rows


# ──────────────────────────────────────────────
# LEN VERI SETI BILGISI
# ──────────────────────────────────────────────

LEN_INFO = """
LEN (Large Engagement Networks) veri seti:
- URL: https://erdemub.github.io/large-engagement-network/
- Icerik: Twitter etkilesim aglari (graph) - metin bazli degil
- Boyut: 170 kampanya + 135 kampanya disi graf
- Ortalama: ~11K dugum, ~23K kenar / graf
- Veri tipi: Graph siniflandirmasi icin uygun, metin analizi icin dogrudan kullanilamaz
- Bu proje metin bazli anlati tespiti yaptigi icin sentetik veri ile devam ediliyor.

MiDe22 veri seti:
- URL: https://huggingface.co/datasets/ogozcelik/turkish-fake-news-detection
- Icerik: 5.064 Turkce tweet (yalan/gercek/oteki etiketli)
- Not: HuggingFace uzerinden tweet metinlerine erisilebilir
- Bu projede kullanimi: Ileri fazlarda degerlendirilecek
"""


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "synthetic_data.csv")
    generate_dataset(output_path)
    print("\n" + LEN_INFO)
