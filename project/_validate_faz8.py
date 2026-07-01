"""Faz 8 dogrulama: HTRI yeni formulle kampanya > organik siralamasi."""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.nlp_pipeline import run_pipeline

df = pd.read_csv('data/synthetic_data.csv', parse_dates=['timestamp', 'account_created_at'])
df['repost_of'] = df['repost_of'].fillna('').astype(str)
df['mentions'] = df['mentions'].fillna('').astype(str)

labels, tsne, summaries = run_pipeline(df, eps=0.45, min_samples=2)

print("=" * 80)
print("FAZ 8 — HTRI Cifte Sayim Duzeltmesi Dogrulama")
print("Formul: 0.30*Coord + 0.25*MediaReuse + 0.25*Accel + 0.20*AgeRisk")
print("Coord  = 0.40*Sync + 0.35*TextAcct + 0.25*(1-Diversity)")
print("=" * 80)

print(f"\nToplam kume: {len([s for s in summaries if not s['is_noise']])}")
print()

# Siralama
sorted_s = sorted(summaries, key=lambda x: x['htri'], reverse=True)

print(f"{'Kume':>4} {'Label':<35} {'HTRI':>6} {'Risk':<8} {'Coord':>6} {'MediaR':>6} {'Accel':>6} {'AgeR':>6} {'Posts':>5} {'Accts':>4}")
print("-" * 95)
for s in sorted_s[:12]:
    cid = s['cluster_id']
    label = s['label']
    htri = s['htri']
    risk = s['risk_level']
    coord = s['coordination_score']
    mr = s['media_reuse_score']
    accel = s['acceleration']
    age = s['age_risk']
    posts = s['post_count']
    accts = s['unique_accounts']
    print(f"{cid:>4} {label:<35} {htri:>6.3f} {risk:<8} {coord:>6.3f} {mr:>6.3f} {accel:>6.3f} {age:>6.3f} {posts:>5} {accts:>4}")

# Kontrol: kampanya kumeleri organikten yuksek mi?
kampanya_ids = set()
for s in summaries:
    label_lower = s['label'].lower()
    if 'deprem' in label_lower or 'elektrik' in label_lower or 'kesinti' in label_lower or 'siber' in label_lower:
        kampanya_ids.add(s['cluster_id'])

kampanya_htri = [s['htri'] for s in summaries if s['cluster_id'] in kampanya_ids]
organik_htri = [s['htri'] for s in summaries if s['cluster_id'] not in kampanya_ids and not s['is_noise']]

if kampanya_htri:
    print(f"\nKampanya kumeleri HTRI: {[round(h, 3) for h in sorted(kampanya_htri, reverse=True)]}")
if organik_htri:
    print(f"Organik kumeler HTRI:   {[round(h, 3) for h in sorted(organik_htri, reverse=True)]}")

min_kamp = min(kampanya_htri) if kampanya_htri else 0
max_org = max(organik_htri) if organik_htri else 1

if min_kamp > max_org:
    print(f"\nBASARILI: En dusuk kampanya HTRI ({min_kamp:.3f}) > en yuksek organik HTRI ({max_org:.3f})")
else:
    print(f"\nUYARI: Kampanya/organik ayrimi tam degil (min_kamp={min_kamp:.3f}, max_org={max_org:.3f})")
    overlap = [h for h in organik_htri if h > min_kamp]
    if overlap:
        print(f"  {len(overlap)} organik kume kampanya esigini asiyor: {[round(h, 3) for h in sorted(overlap, reverse=True)]}")

# Risk seviyesi dagilimi
risk_count = {}
for s in summaries:
    if not s['is_noise']:
        risk_count[s['risk_level']] = risk_count.get(s['risk_level'], 0) + 1
print(f"\nRisk dagilimi: {risk_count}")

print("\nDogrulama tamam.")
