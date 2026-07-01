"""
Faz 10 — Kucuk LLM ile Anlati Etiketleme

SADECE etiket/baslik uretir. Skorlamaya KARISMAZ.
Hata/gecikme durumunda TF-IDF etiketine geri doner (fallback ZORUNLU).
"""
import os
import re
import warnings
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
    return _model, _tokenizer


def _clean_output(text: str) -> str:
    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r'^assistant\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip('.,;:!?,- ')
    words = text.split()
    if len(words) > 6:
        text = ' '.join(words[:6]).rstrip('.,;:!?,- ')
    return text


def generate_label(
    keywords: str,
    example_posts: list[str],
    max_retries: int = 1,
) -> str:
    """LLM ile anlati basligi uret, basarisizsa orijinal keywords'u dondur."""
    try:
        model, tokenizer = _load_model()
    except Exception as e:
        print(f"[LLM] Model yukleme hatasi: {e}")
        return keywords

    examples_text = "\n".join(f"- {p[:150]}" for p in example_posts[:3])

    messages = [
        {
            "role": "system",
            "content": (
                "Sen bir metin analiz asistanisin. Verilen anahtar kelimeler ve "
                "ornek gonderilere dayanarak, bu anlatiyi en fazla 6 kelimeyle "
                "ozetleyen bir Turkce baslik uret. Sadece basligi yaz, "
                "baska aciklama ekleme."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Anahtar kelimeler: {keywords}\n\n"
                f"Ornek gonderiler:\n{examples_text}\n\n"
                f"Baslik (en fazla 6 kelime):"
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    for attempt in range(max_retries + 1):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Baslik (en fazla 6 kelime):" in response:
                response = response.split("Baslik (en fazla 6 kelime):")[-1].strip()
            else:
                response = response[len(prompt):].strip()

            cleaned = _clean_output(response)
            if cleaned and len(cleaned) > 2:
                return cleaned

        except Exception as e:
            print(f"[LLM] Uretim hatasi (attempt {attempt+1}): {e}")
            if attempt == max_retries:
                return keywords

    return keywords


def generate_all_labels(
    summaries: list[dict],
    df,
    max_clusters: int = 8,
) -> dict[int, str]:
    """Tek bir LLM cagrisi ile tum kumelerin etiketlerini uret.
    max_clusters: kac kumeye LLM etiketi uygulanacak (buyukten kucuge).
    Kalan kumeler TF-IDF etiketiyle kalir."""
    try:
        model, tokenizer = _load_model()
    except Exception as e:
        print(f"[LLM] Model yukleme hatasi: {e}")
        return {}

    # En buyuk kumeleri sec (noise haric)
    clusters = [s for s in summaries if not s["is_noise"]]
    clusters.sort(key=lambda s: s["post_count"], reverse=True)
    clusters = clusters[:max_clusters]

    # Batch prompt olustur
    lines = []
    for s in clusters:
        cid = s["cluster_id"]
        mask = df["cluster_id"] == cid
        examples = df[mask]["text"].head(2).tolist()
        examples_text = " | ".join(e[:120] for e in examples)
        kw = s["label"][:80]
        lines.append(f"Kume {cid}: anahtar_kelimeler={kw}, ornek_metinler={examples_text}")

    batch_input = "\n\n".join(lines)
    messages = [
        {
            "role": "system",
            "content": (
                "Her kume icin en fazla 6 kelimelik Turkce baslik uret. "
                "Ornek format:\n"
                "Kume 0: Deprem Uyari Sistemi\n"
                "Kume 1: Elektrik Kesintisi Protestosu\n"
            ),
        },
        {
            "role": "user",
            "content": batch_input + "\n\nKume basliklari:",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Yaniti prompt'tan ayir
        response = re.split(r'assistant\s*', response, flags=re.IGNORECASE)[-1].strip()
        response = response.split("Kume basliklari:")[-1].strip() if "Kume basliklari:" in response else response

        # Her kume icin ciktiyi parse et
        label_map = {}
        for s in clusters:
            cid = s["cluster_id"]
            # Ara: "Kume X: baslik" deseni
            pat = rf'Kume\s*{cid}\s*[:\-–]+\s*(.+?)(?:\n|Kume\s|\Z)'
            m = re.search(pat, response, re.DOTALL)
            if m:
                cleaned = _clean_output(m.group(1))
                if cleaned and len(cleaned) > 2 and cleaned != s["label"][:len(cleaned)]:
                    label_map[cid] = cleaned

        if label_map:
            print(f"[LLM] Batch uretim: {len(label_map)}/{len(clusters)} kume etiketlendi")
            return label_map

    except Exception as e:
        print(f"[LLM] Batch uretim hatasi: {e}")

    # Fallback: tek tek uret (sadece ilk 5 kume)
    print("[LLM] Batch basarisiz, tek tek deneniyor...")
    label_map = {}
    for s in clusters[:5]:
        cid = s["cluster_id"]
        mask = df["cluster_id"] == cid
        examples = df[mask]["text"].head(3).tolist()
        label = generate_label(s["label"], examples, max_retries=0)
        if label != s["label"]:
            label_map[cid] = label
    return label_map
