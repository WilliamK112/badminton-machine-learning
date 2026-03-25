import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
in_csv = ROOT / 'data' / 'quant_features_v2.csv'
out_csv = ROOT / 'data' / 'quant_features_v3.csv'

df = pd.read_csv(in_csv).sort_values('frame').reset_index(drop=True)

# Racket proxy features from wrist/elbow surrogate motion already embedded in forearm angles.
# Add temporal deltas as racket-swing proxies.
for side in ['X', 'Y']:
    fa = f'{side}_r_forearm'
    fb = f'{side}_l_forearm'
    df[f'{side}_racket_proxy_dir'] = (df[fa] - df[fb]).fillna(0)
    df[f'{side}_racket_proxy_speed'] = df[f'{side}_racket_proxy_dir'].diff().fillna(0).abs()

# shuttle acceleration proxy
df['shuttle_accel'] = df['shuttle_speed'].diff().fillna(0)

df.to_csv(out_csv, index=False)
print('saved', out_csv)
