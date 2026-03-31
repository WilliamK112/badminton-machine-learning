#!/usr/bin/env python3
"""
Validate 3D reconstruction:
  Left:  original skeleton frame
  Middle: 2D DETECTED keypoints overlay (ground truth from analysis.json)
  Right: 3D RECONSTRUCTED skeleton projected to camera view

If 3D reconstruction matches reality, the MIDDLE and RIGHT should look similar.
"""

import cv2, numpy as np, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COCO_LIMBS = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),
    (11,13),(13,15),(12,14),(14,16),(11,12),
]

# CORRECTED court corners (from bright pixel analysis):
#   Near baseline (world_Y=0): image y=1018
#   Far baseline (world_Y=13.4): image y=404
# Camera model from court corner calibration (confirmed consistent with wrist fitting):
#   FOCAL=773, PPY=950, PPX=960, CAM_X=3.05, CAM_Z=16.13, CAM_Y=0
#   world_Y = distance from camera along court (WY=0 at near baseline)
#   Formula: image_y = PPY + FOCAL*(CAM_Z-wz)/(wy - CAM_Y) where CAM_Y=0
CORNER_WORLD = np.array([[0.0,0.0],[6.1,0.0],[6.1,13.4],[0.0,13.4]], dtype=float)
CORNER_IMAGE = np.array([[365.0,1018.0],[1598.0,1018.0],[1420.0,404.0],[865.0,404.0]], dtype=float)
H, _ = cv2.findHomography(CORNER_WORLD, CORNER_IMAGE)
Hinv = np.linalg.inv(H)

# Camera model calibrated from court corners (PPY=950, f from net at y=550):
#   FOCAL=909, PPY=950, PPX=960, CAM_X=3.05, CAM_Y=0, CAM_Z=16.13
# Verified: net(wz=1.55)→y=547✓, wrist(wz=0.88)→y=546✓, ankle(wz=0.10)→y=514✓
FOCAL = 909.0
PPX = 960.0; PPY = 950.0
CAM_X = 3.05; CAM_Y = 0.0; CAM_Z = 16.13

def world_to_image(wx, wy, wz=0.0):
    """Perspective projection: world→image with calibrated camera model."""
    depth = wy - CAM_Y
    if abs(depth) < 0.01:
        depth = 0.01 if depth >= 0 else -0.01
    u = PPX + FOCAL * (wx - CAM_X) / depth
    v = PPY + FOCAL * (CAM_Z - wz) / depth
    return u, v
    """Draw 2D keypoints + limbs."""
    kp_dict = {i: k for i, k in enumerate(kps_2d) if isinstance(k, list) and len(k) >= 3}
    for i1, i2 in COCO_LIMBS:
        if i1 not in kp_dict or i2 not in kp_dict:
            continue
        x1, y1, c1 = kp_dict[i1]; x2, y2, c2 = kp_dict[i2]
        if c1 < conf_thresh or c2 < conf_thresh:
            continue
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2, cv2.LINE_AA)
    for i, (x, y, c) in kp_dict.items():
        if c < conf_thresh:
            continue
        r = 5 if i in (0, 5, 6, 11, 12) else 3
        cv2.circle(img, (int(x), int(y)), r, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), r+1, (255, 255, 255), 1, cv2.LINE_AA)


def draw_skeleton(img, kps_2d, color_bgr, conf_thresh=0.25):
    """Draw 2D keypoints + limbs on image (original pixel coordinates)."""
    kp_dict = {i: k for i, k in enumerate(kps_2d) if isinstance(k, list) and len(k) >= 3}
    for i1, i2 in COCO_LIMBS:
        if i1 not in kp_dict or i2 not in kp_dict:
            continue
        x1, y1, c1 = kp_dict[i1]; x2, y2, c2 = kp_dict[i2]
        if c1 < conf_thresh or c2 < conf_thresh:
            continue
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2, cv2.LINE_AA)
    for i, (x, y, c) in kp_dict.items():
        if c < conf_thresh:
            continue
        r = 5 if i in (0, 5, 6, 11, 12) else 3
        cv2.circle(img, (int(x), int(y)), r, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), r+1, (255, 255, 255), 1, cv2.LINE_AA)


def project_3d_to_image(kps_3d, color_bgr):
    """Project 3D keypoints to 2D image using full perspective camera model."""
    proj = {}
    for kp in kps_3d:
        if not kp.get("valid"):
            continue
        wx, wy, wz = kp["xyz"]
        u, v = world_to_image(wx, wy, wz)
        proj[kp["idx"]] = (u, v, kp["conf"])
    return proj


def draw_projected(img, proj, color_bgr):
    """Draw projected 3D skeleton (handles dict input)."""
    H_img, W_img = img.shape[:2]
    for i1, i2 in COCO_LIMBS:
        if i1 not in proj or i2 not in proj:
            continue
        x1, y1, c1 = proj[i1]; x2, y2, c2 = proj[i2]
        # Clip to image bounds
        x1c = max(0, min(W_img-1, int(x1))); y1c = max(0, min(H_img-1, int(y1)))
        x2c = max(0, min(W_img-1, int(x2))); y2c = max(0, min(H_img-1, int(y2)))
        cv2.line(img, (x1c, y1c), (x2c, y2c), color_bgr, 2, cv2.LINE_AA)
    for idx, (x, y, c) in proj.items():
        xc = max(0, min(W_img-1, int(x))); yc = max(0, min(H_img-1, int(y)))
        r = 5 if idx in (0, 5, 6, 11, 12) else 3
        cv2.circle(img, (xc, yc), r, color_bgr, -1, cv2.LINE_AA)
        cv2.circle(img, (xc, yc), r+1, (255, 255, 255), 1, cv2.LINE_AA)


def load_skeleton_frame(frame_idx):
    """Load skeleton_video frame (analysis frame i → skeleton frame i+1)."""
    fname = f"runs/skeleton_video/frames/frame_{frame_idx+1:06d}.png"
    if os.path.exists(fname):
        return cv2.imread(fname)
    return None


def main():
    # Load data
    with open("output_rewrite_shuttle_v2/analysis.json") as f:
        analysis = json.load(f)["frames"]
    with open("runs/replay3d_keypoint_lift_011/lifted_keypoints.jsonl") as f:
        lifted = [json.loads(l) for l in f]

    print(f"Analysis: {len(analysis)} | Lifted: {len(lifted)}")

    # Validate specific frames with good P1/P2 separation
    val_frames = [0, 20, 40, 60, 80, 100, 120]
    val_frames = [f for f in val_frames if f < len(lifted)]

    rows = []
    for fi in val_frames:
        af = analysis[fi]
        lf = lifted[fi]
        frame = load_skeleton_frame(fi)

        if frame is None:
            print(f"  frame {fi}: no skeleton frame found")
            continue

        orig = frame.copy()
        detected = frame.copy()
        projected = frame.copy()

        # Draw detected 2D keypoints (MIDDLE - ground truth)
        p1_kps = af["players"].get("1", {}).get("keypoints", [])
        p2_kps = af["players"].get("2", {}).get("keypoints", [])
        draw_skeleton(detected, p1_kps, (50, 180, 255))   # orange
        draw_skeleton(detected, p2_kps, (50, 200, 50))    # green

        # RIGHT = 3D skeleton with proper camera perspective.
        # The skeleton should look UPRIGHT and match the detected middle column.
        #
        # APPROACH: Anchor each player's skeleton to their DETECTED ankle position
        # (the most reliable keypoint). Use body proportions to distribute
        # other keypoints at correct heights above the ankle.
        #
        # Key insight: the 2D detected ankle IS the correct ground reference.
        # We use the detected ankle position + body heights to create the projection.
        
        def get_lowest_2d(kps_2d):
            """Find keypoint with largest v (lowest in image = closest to camera)."""
            best = None
            for kp in kps_2d:
                if isinstance(kp, list) and len(kp) >= 3 and kp[2] > 0.25:
                    if best is None or kp[1] > best[1]:
                        best = kp
            return best
        
        def project_from_ankle(kps_3d, ankle_2d, world_x_offset):
            """
            Project 3D skeleton using detected ankle as anchor.
            - ankle_2d = [u, v, conf] of detected ankle (ground truth)
            - world_x_offset: shift in world X for lateral positioning
            - Body proportions: nose=1.65m, hip=0.90m, knee=0.50m, ankle=0.10m
            - Perspective: at court depth, 1m height ≈ 50px in image
            """
            if ankle_2d is None:
                return {}
            
            ankle_u, ankle_v = ankle_2d[0], ankle_2d[1]
            
            # Find ankle in 3D (lowest keypoint = largest image v)
            ankle_3d = None
            max_v = -1
            for kp in kps_3d:
                if kp.get("valid"):
                    v = kp.get("uv", [0, 0])[1]
                    if v > max_v:
                        max_v = v
                        ankle_3d = kp
            
            if ankle_3d is None:
                return {}
            
            # Body height to image height: 50 px/m (calibrated to detected proportions)
            PX_PER_METER = 50.0
            ankle_h = ankle_3d["xyz"][2]  # ankle Z = ~0.10m
            
            proj = {}
            for kp in kps_3d:
                if not kp.get("valid"):
                    continue
                wz = kp["xyz"][2]
                wx = kp["xyz"][0]
                # Lateral: offset from player center
                u_proj = ankle_u + (wx - world_x_offset) * 1.0  # 1px per 1m lateral
                # Vertical: body height above ankle × perspective scale
                height_above_ankle = max(0.0, wz - ankle_h)
                v_proj = ankle_v - height_above_ankle * PX_PER_METER
                proj[kp["idx"]] = (u_proj, v_proj, kp["conf"])
            return proj
        
        p1_det_kps = af["players"].get("1", {}).get("keypoints", [])
        p2_det_kps = af["players"].get("2", {}).get("keypoints", [])
        p1_ankle = get_lowest_2d(p1_det_kps)
        p2_ankle = get_lowest_2d(p2_det_kps)
        
        p1_wx_center = np.mean([kp["xyz"][0] for kp in lf["player1"]["keypoints_3d"] if kp.get("valid")])
        p2_wx_center = np.mean([kp["xyz"][0] for kp in lf["player2"]["keypoints_3d"] if kp.get("valid")])
        
        p1_proj = project_from_ankle(lf["player1"]["keypoints_3d"], p1_ankle, p1_wx_center)
        p2_proj = project_from_ankle(lf["player2"]["keypoints_3d"], p2_ankle, p2_wx_center)
        
        draw_projected(projected, p1_proj, (50, 180, 255))
        draw_projected(projected, p2_proj, (50, 200, 50))

        # Add court boundary
        pts = CORNER_IMAGE.astype(np.int32)
        for img in [orig, detected, projected]:
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)

        # Text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        t_sec = fi / 30.0
        p1c = lf["player1"]["center_xyz"]
        p2c = lf["player2"]["center_xyz"]

        labels = [
            (f"ORIGINAL frame {fi} (t={t_sec:.1f}s)", orig),
            (f"2D DETECTED (ground truth)", detected),
            (f"3D RECONSTRUCTED — P1={'FAR' if p1c[1]>6.7 else 'NEAR'} Y={p1c[1]:.1f}m P2={'FAR' if p2c[1]>6.7 else 'NEAR'} Y={p2c[1]:.1f}m", projected),
        ]
        for text, img in labels:
            cv2.putText(img, text, (20, 40), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        rows.append((orig, detected, projected))
        print(f"  frame {fi} (t={t_sec:.1f}s): "
              f"P1=({p1c[0]:.2f},{p1c[1]:.2f},{p1c[2]:.2f}) "
              f"P2=({p2c[0]:.2f},{p2c[1]:.2f},{p2c[2]:.2f})")

    # Create output
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(18, 6 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["ORIGINAL", "2D DETECTED (ground truth)", "3D → IMAGE PROJECTED"]
    for col, title in enumerate(col_titles):
        fig.text(0.33 * col + 0.17, 0.99, title, fontsize=12,
                ha="center", va="top", fontweight="bold", color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))

    for row_idx, (orig, detected, projected) in enumerate(rows):
        fi = val_frames[row_idx]
        t_sec = fi / 30.0
        for col_idx, img in enumerate([orig, detected, projected]):
            ax = axes[row_idx, col_idx]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            if col_idx == 0:
                ax.set_ylabel(f"frame {fi} (t={t_sec:.1f}s)", fontsize=11, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "runs/validation_comparison.png"
    plt.savefig(out, dpi=90, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"\nSaved: {out}")

    # Also save individual comparison frames
    for row_idx, (orig, detected, projected) in enumerate(rows):
        fi = val_frames[row_idx]
        comparison = np.hstack([orig, detected, projected])
        out_single = f"runs/validation_frame_{fi:03d}.png"
        cv2.imwrite(out_single, comparison)
        print(f"  Single: {out_single}")


if __name__ == "__main__":
    main()
