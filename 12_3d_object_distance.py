"""
Object distance per bbox from PCD — with bbox shrink + styled distance label.

Output visualization matches reference image style:
- Green rounded box above bbox
- White bold distance text (e.g. 2.34m)
- Black shadow for readability
- Center dot inside bbox
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from typing import Tuple, Optional, Dict, List

# -------- PATHS --------
BASE_DIR   = Path.cwd()
RAW_DIR    = BASE_DIR / "raw/test"
RGB_DIR    = RAW_DIR / "camera_color_image_raw"
PAIRS_CSV  = RAW_DIR / "rgb_sensor_pairs.csv"
IR_CSV     = BASE_DIR / "runs/ir_filter/ir_filter_manual.csv"
PCD_ROOT   = RAW_DIR / "camera_depth_points"
CAM_ROOT   = RAW_DIR / "camera_depth_camera_info"

OUT_DIR    = BASE_DIR / "runs/object_distance"
OUT_IMG    = OUT_DIR / "images"
OUT_CSV    = OUT_DIR / "object_distance.csv"


# -------- PARAMS --------
BBOX_SHRINK = 5        # px shrink — SAME AS YOUR EXAMPLE  ✔
KNN_K_MAX = 5         
FONT      = cv2.FONT_HERSHEY_SIMPLEX
LABEL_BG  = (30, 200, 30)    # green like your sample
LABEL_FG  = (255, 255, 255)  # white text
LABEL_SHADOW = (0, 0, 0)     # black outline


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    OUT_IMG.mkdir(parents=True, exist_ok=True)


def load_ir_boxes(ir_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ir_csv)
    return df[df["decision"] == "ir_real_object"].copy()


def pick_paths_from_pairs(img_name: str):
    df = pd.read_csv(PAIRS_CSV)
    if "rgb_path" not in df.columns:
        return None, None

    rows = df[df["rgb_path"].astype(str).str.endswith(img_name)]
    if rows.empty:
        return None, None

    row = rows.iloc[0]
    pcd_path = cam_path = None

    for val in map(str, row.values):
        lv = val.lower()
        if lv.endswith(".pcd"):
            p = (RAW_DIR / val).resolve()
            if p.exists(): pcd_path = p
        if lv.endswith(".txt") and "camera_depth_info" in lv:
            p = (RAW_DIR / val).resolve()
            if p.exists(): cam_path = p

    return pcd_path, cam_path


def parse_caminfo_txt(path: Path):
    fx=fy=cx=cy=None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("K:"):
                arr = line.split("[",1)[1].split("]",1)[0]
                vals = [float(x) for x in arr.split(",")]
                fx, cx = vals[0], vals[2]
                fy, cy = vals[4], vals[5]
                break
    return fx, fy, cx, cy


def read_pcd_ascii_xyz(path: Path):
    fields=[]; lines=[]; in_data=False
    with path.open("r") as f:
        for s in f:
            s=s.strip()
            if not in_data:
                if s.startswith("FIELDS"):
                    fields=s.split()[1:]
                elif s.startswith("DATA"):
                    in_data=True
                continue
            if s: lines.append(s)

    arr = np.loadtxt(lines, float)
    if arr.ndim==1: arr = arr[None,:]
    col={n:i for i,n in enumerate(fields)}
    return arr[:,col["x"]], arr[:,col["y"]], arr[:,col["z"]]


def project_points(X,Y,Z, fx,fy,cx,cy, W,H):
    m = (Z>0) & np.isfinite(Z)
    X,Y,Z = X[m],Y[m],Z[m]

    u = fx*(X/Z)+cx
    v = fy*(Y/Z)+cy
    m2 = (u>=0)&(u<W)&(v>=0)&(v<H)
    return u[m2], v[m2], X[m2], Y[m2], Z[m2]


def knn_center(u,v, idx, x1,y1,x2,y2, k):
    uc, vc = (x1+x2)/2, (y1+y2)/2
    du = u[idx]-uc
    dv = v[idx]-vc
    k=min(k, len(idx))
    sel = np.argpartition(du*du+dv*dv, k-1)[:k]
    return idx[sel]


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def draw_distance_label(img, x1,y1,x2,y2, dist_m):
    """
    Draw a green rounded rectangle label above the bbox (matching your sample).
    """

    txt = f"{dist_m:.2f}m"
    scale = 0.9
    thickness = 2
    (tw, th), bl = cv2.getTextSize(txt, FONT, scale, thickness)

    pad = 6
    bx1 = x1
    by1 = y1 - th - pad*2 - 8
    if by1 < 0: by1 = 0

    bx2 = x1 + tw + pad*2
    by2 = y1 - 4

    # background rectangle
    cv2.rectangle(img, (bx1,by1), (bx2,by2), LABEL_BG, -1, cv2.LINE_AA)

    # shadow text
    cv2.putText(img, txt, (bx1+pad, by2-pad-2), FONT, scale, LABEL_SHADOW, 4, cv2.LINE_AA)

    # white text
    cv2.putText(img, txt, (bx1+pad, by2-pad-2), FONT, scale, LABEL_FG, 2, cv2.LINE_AA)


def draw_center_dot(img, x1,y1,x2,y2):
    cx = int((x1+x2)/2)
    cy = int((y1+y2)/2)
    cv2.circle(img, (cx,cy), 4, (0,255,0), -1, cv2.LINE_AA)  # green center dot


# ============================================================
# PER-IMAGE PROCESSING
# ============================================================

def process_image(img_name, grp):
    out_rows = []

    rgb = cv2.imread(str(RGB_DIR/img_name))
    if rgb is None:
        return []

    H,W = rgb.shape[:2]
    vis = rgb.copy()

    pcd_path, caminfo_path = pick_paths_from_pairs(img_name)
    if not pcd_path or not caminfo_path:
        return []

    fx,fy,cx,cy = parse_caminfo_txt(caminfo_path)
    X,Y,Z = read_pcd_ascii_xyz(pcd_path)
    u,v,Xv,Yv,Zv = project_points(X,Y,Z, fx,fy,cx,cy, W,H)

    for _,r in grp.iterrows():
        x1,y1,x2,y2 = map(int, (r.x1,r.y1,r.x2,r.y2))

        # --- SHRINK (same as your example) --------------------------
        x1 = max(0, x1 + BBOX_SHRINK)
        y1 = max(0, y1 + BBOX_SHRINK)
        x2 = min(W-1, x2 - BBOX_SHRINK)
        y2 = min(H-1, y2 - BBOX_SHRINK)
        # ------------------------------------------------------------

        # points inside
        m = (u>=x1)&(u<=x2)&(v>=y1)&(v<=y2)
        idx = np.where(m)[0]
        if len(idx)==0:
            continue

        idx = knn_center(u, v, idx, x1,y1,x2,y2, KNN_K_MAX)

        Zi = Zv[idx]
        Xi,Yi = Xv[idx], Yv[idx]
        rng = np.sqrt(Xi*Xi + Yi*Yi + Zi*Zi)

        depth_z = float(np.median(Zi))
        range_m = float(np.median(rng))

        # ---- VISUALIZATION ----
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        draw_center_dot(vis, x1,y1,x2,y2)
        draw_distance_label(vis, x1,y1,x2,y2, range_m)

        out_rows.append(dict(
            image=img_name,
            x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2,
            depth_z_m=round(depth_z,3),
            range_m=round(range_m,3),
            n_points=len(idx),
            status="ok"
        ))

    cv2.imwrite(str(OUT_IMG / img_name), vis)
    return out_rows


# ============================================================
# MAIN RUN
# ============================================================

def run():
    ensure_dirs()
    boxes = load_ir_boxes(IR_CSV)
    all_rows = []

    for img_name, grp in boxes.groupby("image"):
        all_rows.extend(process_image(img_name, grp))

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"Saved CSV:   {OUT_CSV}")
    print(f"Saved images:{OUT_IMG}")


if __name__ == "__main__":
    run()
