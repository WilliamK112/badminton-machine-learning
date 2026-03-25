import json
from pathlib import Path
import math
import csv

ROOT = Path(__file__).resolve().parents[1]
feat_file = ROOT / "data" / "frame_features_v2.jsonl"
out_csv = ROOT / "data" / "quant_features_v2.csv"
rows = [json.loads(x) for x in feat_file.read_text().splitlines() if x.strip()]

def angle(a,b,c):
    ab=(a[0]-b[0],a[1]-b[1]); cb=(c[0]-b[0],c[1]-b[1])
    nab=(ab[0]**2+ab[1]**2)**0.5; ncb=(cb[0]**2+cb[1]**2)**0.5
    if nab==0 or ncb==0: return 0.0
    dot=ab[0]*cb[0]+ab[1]*cb[1]
    x=max(-1,min(1,dot/(nab*ncb)))
    return math.degrees(math.acos(x))

def get_angles(k):
    p=[(pt[0],pt[1]) for pt in k]
    return {
      'l_forearm': angle(p[5],p[7],p[9]) if len(p)>9 else 0,
      'r_forearm': angle(p[6],p[8],p[10]) if len(p)>10 else 0,
      'l_upperarm': angle(p[11],p[5],p[7]) if len(p)>11 else 0,
      'r_upperarm': angle(p[12],p[6],p[8]) if len(p)>12 else 0,
      'torso_rot': abs((p[6][0]-p[5][0])-(p[12][0]-p[11][0])) if len(p)>12 else 0,
      'l_thigh': angle(p[5],p[11],p[13]) if len(p)>13 else 0,
      'r_thigh': angle(p[6],p[12],p[14]) if len(p)>14 else 0,
      'l_calf': angle(p[11],p[13],p[15]) if len(p)>15 else 0,
      'r_calf': angle(p[12],p[14],p[16]) if len(p)>16 else 0,
    }

fields = [
  'frame','t_sec','winner_proxy','shuttle_x','shuttle_y','shuttle_speed',
  'X_l_forearm','X_r_forearm','X_l_upperarm','X_r_upperarm','X_torso_rot','X_l_thigh','X_r_thigh','X_l_calf','X_r_calf',
  'Y_l_forearm','Y_r_forearm','Y_l_upperarm','Y_r_upperarm','Y_torso_rot','Y_l_thigh','Y_r_thigh','Y_l_calf','Y_r_calf'
]

with out_csv.open('w', newline='') as f:
    wr=csv.DictWriter(f, fieldnames=fields)
    wr.writeheader()
    for r in rows:
        row={'frame':r['frame'],'t_sec':r['t_sec']}
        sh=r.get('shuttle',{})
        xy=sh.get('xy') or [0.5,0.5]
        row['shuttle_x']=xy[0]; row['shuttle_y']=xy[1]; row['shuttle_speed']=sh.get('speed',0)
        row['winner_proxy']=1 if xy[1]>0.5 else 0
        for side in ['X','Y']:
            p=r.get('players',{}).get(side)
            a=get_angles(p['kpts']) if p and p.get('kpts') else {k:0 for k in ['l_forearm','r_forearm','l_upperarm','r_upperarm','torso_rot','l_thigh','r_thigh','l_calf','r_calf']}
            for k,v in a.items(): row[f'{side}_{k}']=v
        wr.writerow(row)

print('saved', out_csv)
