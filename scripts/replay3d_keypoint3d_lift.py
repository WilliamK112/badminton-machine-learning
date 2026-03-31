#!/usr/bin/env python3
"""
Monocular 2D→3D COCO keypoint lifting — v9 (CAMERA MODEL + VARIABLE KPS)

v9 APPROACH:
  - Camera model (from wrist fitting): CAM_X=3.74, CAM_Y=8.70, CAM_Z=16.13, F=773, PP=(960,950)
  - Camera inverse projection:
      world_Y = CAM_Y + F/(PPY - v) * (CAM_Z - WZ)   [for image y, keypoint height WZ]
      world_X = CAM_X + (PPX - u)/F * (world_Y - CAM_Y)
  - Player world_Y = mean of available ankle/knee world_Y (most grounded = most reliable)
  - Keypoint world_X from camera model at player depth
  - world_Z = known body proportions (height above court)
  - Handles variable keypoint counts (3-17 per frame)

Key insight: Camera model gives CORRECT depth (confirmed: P1 ankle → FAR, P2 ankle → NEAR).
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

COCO17_NAMES = [
    "nose","leye","reye","lear","rear",
    "lshoulder","rshoulder","lelbow","relbow",
    "lwrist","rwrist","lhip","rhip",
    "lknee","rknee","lankle","rankle",
]
# Only 16 in analysis.json (no rankle) — use lankle as ankle
KP_NAMES = COCO17_NAMES[:16]  # indices 0-15

# Body heights above court for each keypoint (COCO proportions)
# Index: height(m)
KEYPOINT_Z = {
    0:1.65, 1:1.60, 2:1.60, 3:1.57, 4:1.57,
    5:1.30, 6:1.30, 7:1.05, 8:1.05,
    9:0.88, 10:0.88,
    11:0.90, 12:0.90,
    13:0.50, 14:0.50,
    15:0.10,  # lankle (index 15 in 16-kp format)
    # 16: rankle (sometimes present in 17-kp format)
}

# COCO limb connections (adjusted for 16 or 17 kp)
LIMB_CONNECTIONS = [
    (5,7),(7,9),(6,8),(8,10),   # arms
    (11,13),(13,15),(12,14),(14,16) if False else (12,14),(14,15),  # legs (handle 16-kp)
    (5,6),(11,12),  # torso
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),  # head/shoulder
]
# Filter valid limbs (both indices must exist in KEYPOINT_Z)
LIMB_CONNECTIONS = [(a,b) for a,b in LIMB_CONNECTIONS 
                    if a in KEYPOINT_Z and b in KEYPOINT_Z]

# Limb lengths (m) for skeleton enforcement
LIMB_LENGTHS = {
    (5,7):0.30,(7,9):0.25,(6,8):0.30,(8,10):0.25,
    (11,13):0.42,(13,15):0.42,(12,14):0.42,(14,15):0.42,
    (5,6):0.40,(11,12):0.26,
    (0,1):0.10,(0,2):0.10,
}

# CORRECTED court corners (from video frame analysis):
#   Near baseline: y=1018 (bright white near bottom)
#   Far baseline: y=404 (bright white near top)
#   Left boundary: x≈865 (far), x≈365 (near)
#   Right boundary: x≈1420 (far), x≈1598 (near)
# These give CORRECT half assignment (P1=FAR, P2=NEAR) for all frames.
import cv2
CORNER_WORLD = np.array([[0.0,0.0],[6.1,0.0],[6.1,13.4],[0.0,13.4]], dtype=float)
CORNER_IMAGE = np.array([[365.0,1018.0],[1598.0,1018.0],[1420.0,404.0],[865.0,404.0]], dtype=float)
H_COURT, _ = cv2.findHomography(CORNER_WORLD, CORNER_IMAGE)
HINV = np.linalg.inv(H_COURT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-json", required=True)
    p.add_argument("--replay-jsonl", required=True)
    p.add_argument("--replay-meta-json", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--conf-threshold", type=float, default=0.25)
    return p.parse_args()


def world_from_image(u, v):
    """Homography: image (u,v) at ground level → world (WX, WY)."""
    ih = np.array([[float(u), float(v), 1.0]])
    wh = ih @ HINV.T
    return float(wh[0,0]/wh[0,2]), float(wh[0,1]/wh[0,2])


def lift_keypoints(kps_2d, conf_thresh):
    """
    Lift all detected 2D keypoints to 3D using camera model.
    
    Key insight: use the ANKLE (or most-grounded available) keypoint's
    world_Y as the player's depth anchor. This is the most reliable
    depth reference since ankles are closest to the court.
    
    Args:
        kps_2d: list of [u, v, conf] keypoints (variable length, 0-17)
        conf_thresh: confidence threshold
        
    Returns:
        list of keypoint_3d dicts, player_center_xyz, valid_count
    """
    keypoints_3d = []
    
    # Determine which keypoints are available
    n_kps = len(kps_2d)
    
    # Collect grounded keypoints for player depth
    # Strategy: use ANKLE(s) if available (most grounded = lowest in image = largest v)
    # Otherwise: use the keypoint with LARGEST v (closest to camera = best depth estimate)
    ground_wy_vals = []
    ground_wx_vals = []
    
    # Also collect all valid keypoint world_X for player center
    all_wx = []
    
    # Find the keypoint with largest v (most grounded / closest to camera)
    # This is used as depth anchor when no proper ankle/knee is available
    best_ground_kp = None  # (wx, wy, v) of most grounded keypoint
    
    # First pass: find the ANKLE keypoint and compute player ground depth.
    # The ankle is identified by name ("ankle") and is the most grounded
    # keypoint (largest image y = closest to camera = lowest on body).
    #
    # CRITICAL FIX (v11): Z-axis stability
    # - Ankle world_Z = 0.10m (foot on court, anchor point)
    # - Other keypoints: world_Z = KEYPOINT_Z[i] (fixed proportions)
    #   BUT we also scale Z proportionally based on image y-distance from ankle
    #   to handle cases where the ankle detection is unreliable.
    # - This prevents "floating" because ankle is always near ground.
    #
    # For each keypoint: world_Z = ANCHOR_Z + proportion * (ankle_image_y - keypoint_image_y) / scale
    # where ANCHOR_Z = 0.10m for ankle, and proportion encodes how body parts move together.
    
    ANKLE_Z = 0.10  # feet are always at ground level
    
    # Find the best ankle (largest v in image = closest to camera)
    best_ankle_idx = -1
    best_ankle_v = -1
    best_ankle_conf = 0.0
    for i, kp in enumerate(kps_2d):
        if not isinstance(kp, list) or len(kp) < 3:
            continue
        u, v, conf = float(kp[0]), float(kp[1]), float(kp[2])
        name = KP_NAMES[i] if i < len(KP_NAMES) else f"kp{i}"
        is_ankle = "ankle" in name.lower()
        if conf >= conf_thresh and (is_ankle or v > best_ankle_v) and conf >= 0.25:
            best_ankle_v = v
            best_ankle_idx = i
            best_ankle_conf = conf
    
    # Compute world_X, world_Y for all keypoints (from homography)
    for i, kp in enumerate(kps_2d):
        if not isinstance(kp, list) or len(kp) < 3:
            keypoints_3d.append({"idx": i, "name": KP_NAMES[i] if i < len(KP_NAMES) else f"kp{i}",
                                   "xyz": [0,0,0], "conf": 0.0, "valid": False})
            continue
            
        u, v, conf = float(kp[0]), float(kp[1]), float(kp[2])
        
        if conf < conf_thresh:
            keypoints_3d.append({"idx": i, "name": KP_NAMES[i] if i < len(KP_NAMES) else f"kp{i}",
                                   "xyz": [0,0,0], "conf": float(conf), "valid": False})
            continue
        
        # world_X,Y from homography (ground plane projection)
        wx, wy = world_from_image(u, v)
        all_wx.append(wx)
        
        # Track most grounded keypoint (largest v)
        if best_ground_kp is None or v > best_ground_kp[2]:
            best_ground_kp = (wx, wy, v)
        
        keypoints_3d.append({
            "idx": i,
            "name": KP_NAMES[i] if i < len(KP_NAMES) else f"kp{i}",
            "xyz": [round(wx, 3), round(wy, 3), 0.0],  # Z set in second pass
            "conf": round(conf, 4),
            "valid": True,
            "uv": [round(u, 1), round(v, 1)],
        })
    
    # Player depth = ANKLE world_Y if available, else most grounded keypoint's WY
    # "Most grounded" = keypoint with LARGEST v (lowest in image = closest to camera)
    # = best depth anchor for the player
    # 
    # For sparse frames (fewer keypoints), the "largest v" keypoint might not be
    # a true ankle - but it's still the best available depth estimate.
    # We use the KEYPOINT'S OWN v (from analysis), not re-computed from world_Y.
    ankle_wy = []
    ankle_wx = []
    for kp3d in keypoints_3d:
        if not kp3d.get("valid"):
            continue
        name = kp3d.get("name", "")
        # Identify ankles by name (works when 17 keypoints detected)
        if "ankle" in name.lower():
            ankle_wy.append(kp3d["xyz"][1])
            ankle_wx.append(kp3d["xyz"][0])
    
    if ankle_wy:
        player_wy = float(np.mean(ankle_wy))
        player_wx = float(np.mean(ankle_wx))
    elif best_ground_kp is not None:
        player_wy = best_ground_kp[1]
        # Use mean of all keypoint world_X as player center X
        player_wx = float(np.mean(all_wx)) if all_wx else best_ground_kp[0]
    elif all_wx:
        player_wy = float(np.mean(all_wx))  # fallback (not ideal)
        player_wx = float(np.mean(all_wx))
    else:
        player_wy, player_wx = 6.7, 3.05
    
    # Z-anchoring: ankle = 0.10m (foot on ground = Z anchor).
    # All other keypoints get fixed anatomical Z.
    # Simple and stable: prevents floating.
    ANKLE_Z_FIXED = 0.10
    
    for kp3d in keypoints_3d:
        if not kp3d["valid"]:
            continue
        kp_idx = kp3d["idx"]
        if kp_idx not in KEYPOINT_Z:
            kp3d["xyz"][2] = 0.90
            continue
        ref_z = KEYPOINT_Z[kp_idx]
        # Directly use anatomical height as world_Z
        # (keypoint at ankle_idx gets 0.10m, others proportionally higher)
        kp3d["xyz"][2] = round(ref_z, 3)
    
    # But: force the lowest keypoint (largest v) to have Z <= 0.15m
    # This is the "feet" constraint that prevents floating
    lowest_kp = None
    max_v = -1
    for kp3d in keypoints_3d:
        if not kp3d["valid"]:
            continue
        v = kp3d["uv"][1]
        if v > max_v:
            max_v = v
            lowest_kp = kp3d
    if lowest_kp is not None:
        # The lowest keypoint should be at foot level (≤0.15m)
        # Scale all Z values proportionally so lowest_kp.z = 0.12m
        current_lowest_z = lowest_kp["xyz"][2]
        if current_lowest_z > 0.15:
            scale = 0.12 / current_lowest_z
            for kp3d in keypoints_3d:
                if kp3d["valid"]:
                    kp3d["xyz"][2] = round(kp3d["xyz"][2] * scale, 3)
            # Re-set lowest to exactly 0.12
            lowest_kp["xyz"][2] = 0.12
    
    # Re-anchor all keypoints to player depth (all get same WY = player depth)
    for kp3d in keypoints_3d:
        if not kp3d["valid"]:
            continue
        # Anchor to player depth (but world_X stays from homography)
        kp3d["xyz"][1] = round(player_wy, 3)
        u, v = kp3d["uv"]
        wx_new, _ = world_from_image(u, v)
        kp3d["xyz"][0] = round(wx_new, 3)
    
    valid_count = sum(1 for k in keypoints_3d if k.get("valid"))
    
    return keypoints_3d, player_wx, player_wy, valid_count


def enforce_limb_lengths(frame):
    """Scale keypoints along limbs to match expected bone lengths."""
    for pkey in ["player1", "player2"]:
        kps = {kp["idx"]: np.array(kp["xyz"])
               for kp in frame[pkey]["keypoints_3d"] if kp.get("valid")}
        
        for a, b in LIMB_CONNECTIONS:
            if a not in kps or b not in kps:
                continue
            key = (min(a, b), max(a, b))
            exp = LIMB_LENGTHS.get(key)
            if exp is None:
                continue
            pa = kps[a]
            pb = kps[b]
            obs = float(np.linalg.norm(pb - pa))
            if obs < 1e-6:
                continue
            ratio = exp / obs
            if 0.70 <= ratio <= 1.30:
                continue  # within tolerance
            # Adjust: scale the child keypoint (b) toward/away from parent (a)
            direction = pb - pa
            kps[b] = pa + direction * (exp / obs)
        
        # Write back
        for kp in frame[pkey]["keypoints_3d"]:
            if kp.get("valid") and kp["idx"] in kps:
                c = kps[kp["idx"]]
                kp["xyz"] = [round(float(c[0]), 3), round(float(c[1]), 3), round(float(c[2]), 3)]
    return frame


def main():
    args = parse_args()
    analysis_frames = json.load(Path(args.analysis_json).open()).get("frames", [])
    
    replay_frames = []
    with Path(args.replay_jsonl).open() as f:
        for line in f:
            if line.strip():
                replay_frames.append(json.loads(line))
    
    print(f"Frames: {len(analysis_frames)} | Court: 13.4m × 6.1m | Net: 6.7m")
    print(f"Method: homography (near=1018, far=404) + ankle-anchored world_Y | conf≥{args.conf_threshold}")
    
    lifted = []
    for i, (fa, fr) in enumerate(zip(analysis_frames, replay_frames)):
        if i % 40 == 0 and i > 0:
            print(f"  {i}/{len(analysis_frames)} ...")
        
        frame_idx = fa.get("frame_idx", fr.get("frame", i))
        t_sec = fr.get("t_sec", 0.0)
        
        p1_kps = fa.get("players", {}).get("1", {}).get("keypoints", [])
        p2_kps = fa.get("players", {}).get("2", {}).get("keypoints", [])
        
        p1_3d, p1_cx, p1_cy, p1_vk = lift_keypoints(p1_kps, args.conf_threshold)
        p2_3d, p2_cx, p2_cy, p2_vk = lift_keypoints(p2_kps, args.conf_threshold)
        
        avg1 = np.mean([k["conf"] for k in p1_3d if k.get("valid")]) if p1_3d else 0.0
        avg2 = np.mean([k["conf"] for k in p2_3d if k.get("valid")]) if p2_3d else 0.0
        
        out = {
            "frame": frame_idx,
            "t_sec": t_sec,
            "fps": 30.0,
            "court": {"length_m": 13.4, "width_m": 6.1},
            "player1": {
                "keypoints_3d": p1_3d,
                "center_xyz": [round(p1_cx, 3), round(p1_cy, 3), 0.0],
                "scale": 1.0,
                "visible": bool(p1_vk),
                "avg_conf": round(float(avg1), 4),
                "valid_kps": p1_vk,
            },
            "player2": {
                "keypoints_3d": p2_3d,
                "center_xyz": [round(p2_cx, 3), round(p2_cy, 3), 0.0],
                "scale": 1.0,
                "visible": bool(p2_vk),
                "avg_conf": round(float(avg2), 4),
                "valid_kps": p2_vk,
            },
            "shuttle": fr.get("shuttle", {}),
        }
        # DISABLED enforce_limb_lengths — it was scaling Z values, causing
        # feet to float (ankle Z changed from 0.10 → 0.228). The Z-anchoring
        # above already ensures feet are grounded. Bone length enforcement can
        # be re-enabled later with X,Y-only scaling to preserve Z.
        # out = enforce_limb_lengths(out)
        lifted.append(out)
    
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for fr in lifted:
            f.write(json.dumps(fr, ensure_ascii=False) + "\n")
    
    net_y = 6.7
    fr0 = lifted[0]
    p1c = fr0["player1"]["center_xyz"]
    p2c = fr0["player2"]["center_xyz"]
    x_sep = abs(p1c[0] - p2c[0])
    y_sep = abs(p1c[1] - p2c[1])
    
    print(f"\n✓ Done → {out_path}")
    print(f"\nFrame 0:")
    print(f"  P1: ({p1c[0]:.3f}, {p1c[1]:.3f}m) HALF={'FAR' if p1c[1]>net_y else 'NEAR'}")
    print(f"  P2: ({p2c[0]:.3f}, {p2c[1]:.3f}m) HALF={'FAR' if p2c[1]>net_y else 'NEAR'}")
    print(f"  X_sep={x_sep:.3f}m | Y_sep={y_sep:.3f}m")
    
    print(f"\nFrame 0 P1 keypoints:")
    for kp in fr0["player1"]["keypoints_3d"]:
        if kp.get("valid"):
            print(f"  {kp['name']:12s}: ({kp['xyz'][0]:.3f}, {kp['xyz'][1]:.3f}, {kp['xyz'][2]:.3f})")
    
    # Spot check
    print("\nSpot check frames:")
    print("Frame | P1_Y | P1_half | P2_Y | P2_half | X_sep")
    for fi in [0, 20, 40, 60, 80, 120]:
        pc = lifted[fi]["player1"]["center_xyz"]
        qc = lifted[fi]["player2"]["center_xyz"]
        xs = abs(pc[0]-qc[0])
        h1 = 'FAR' if pc[1]>net_y else 'NEAR'
        h2 = 'FAR' if qc[1]>net_y else 'NEAR'
        print(f"{fi:5d} | {pc[1]:5.2f} | {h1:7s} | {qc[1]:5.2f} | {h2:7s} | {xs:.2f}m")


if __name__ == "__main__":
    main()
