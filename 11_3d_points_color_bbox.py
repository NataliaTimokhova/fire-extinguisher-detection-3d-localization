"""
Overlay all colored registered 3D points on the full RGB image.
Also draw 2D bboxes from labels for visualization.

Inputs:
- raw/test/rgb_sensor_pairs.csv                      (map RGB -> colored PCD + color caminfo)
- runs/ir_filter/ir_filter_manual.csv               (only decision=='ir_real_object' if present)
- raw/test/camera_color_image_raw/*.png
- raw/test/depth_registered_colored_pointclouds/colored_depth_registered_points_*.pcd (ASCII)
- raw/test/camera_color_camera_info/camera_color_info_*.txt    (K line)

Output:
- runs/3d_points_color_bbox/<image_stem>_overlay.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# ---------- PATHS ----------
BASE_DIR = Path.cwd()
RAW_DIR  = BASE_DIR / "raw/test"
RGB_DIR  = RAW_DIR / "camera_color_image_raw"
PAIRS    = RAW_DIR / "rgb_sensor_pairs.csv"
IR_CSV   = BASE_DIR / "runs/ir_filter/ir_filter_manual.csv"

OUT_DIR  = BASE_DIR / "runs/3d_points_color_bbox"

# ---------- PARAMS ----------
DOT_RADIUS       = 1          # crisp dot
DOT_THICKNESS    = -1         # filled
MAX_POINTS_ALL   = 300000     # global subsample if cloud too dense
BBOX_SHRINK_PX   = 2          # avoid borders on drawn boxes only
TRIM_TOP_FRAC    = 0.0        # set to 0.35 if you want to trim
TRIM_BOTTOM_FRAC = 0.0
FONT             = cv2.FONT_HERSHEY_SIMPLEX

# ---------- UTIL ----------
def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_boxes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "decision" in df.columns:
        df = df[df["decision"] == "ir_real_object"].copy()
    for c in ("x1", "y1", "x2", "y2"):
        df[c] = df[c].astype(float).round().astype(int)
    return df

def parse_caminfo_txt(path: Path):
    fx = fy = cx = cy = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("K:"):
                vals = [float(x) for x in s.split("[", 1)[1].split("]", 1)[0].split(",")]
                fx, cx = vals[0], vals[2]
                fy, cy = vals[4], vals[5]
                break
    if None in (fx, fy, cx, cy):
        raise ValueError(f"K not found in {path}")
    return fx, fy, cx, cy

def _unpack_rgb_u32_to_bgr(u32: np.ndarray) -> np.ndarray:
    if u32.dtype == np.float32:  # PCL float-packed rgb
        u32 = u32.view(np.uint32)
    u = u32.astype(np.uint32)
    r = ((u >> 16) & 0xFF).astype(np.uint8)
    g = ((u >> 8)  & 0xFF).astype(np.uint8)
    b = (u & 0xFF).astype(np.uint8)
    return np.stack([b, g, r], axis=1)

def read_colored_pcd_ascii(path: Path):
    fields = []
    types  = []
    rows   = []
    in_data = False
    mode = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for s in f:
            s = s.strip()
            if not in_data:
                if s.lower().startswith("fields "):
                    fields = s.split()[1:]
                elif s.lower().startswith("type "):
                    types = s.split()[1:]
                elif s.lower().startswith("data "):
                    mode = s.split(None, 1)[1].lower()
                    in_data = True
                continue
            if s:
                rows.append(s)

    if mode != "ascii":
        raise ValueError(f"Only ASCII PCD supported: {path}")
    if not all(k in fields for k in ("x", "y", "z")):
        raise ValueError("PCD missing x y z")

    arr = np.loadtxt(rows, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]

    col = {n: i for i, n in enumerate(fields)}

    X = arr[:, col["x"]].astype(np.float32)
    Y = arr[:, col["y"]].astype(np.float32)
    Z = arr[:, col["z"]].astype(np.float32)

    if "rgb" in col or "rgba" in col:
        name = "rgb" if "rgb" in col else "rgba"
        if types and len(types) == len(fields) and types[col[name]].lower() == "f":
            rgb_field = arr[:, col[name]].astype(np.float32)
        else:
            rgb_field = arr[:, col[name]].astype(np.uint32)
        BGR = _unpack_rgb_u32_to_bgr(rgb_field)
    else:
        # fallback gray if no rgb
        g = np.full_like(Z, 180, dtype=np.uint8)
        BGR = np.stack([g, g, g], axis=1)

    return X, Y, Z, BGR

def pick_paths_from_pairs_colored(img_name: str):
    df = pd.read_csv(PAIRS)
    if "rgb_path" not in df.columns:
        return None, None

    rows = df[df["rgb_path"].astype(str).str.endswith(img_name)]
    if rows.empty:
        return None, None

    row = rows.iloc[0]
    pcd_path = cam_path = None

    # prefer colored registered cloud + color caminfo
    for val in map(str, row.values):
        lv = val.lower()
        if lv.endswith(".pcd") and "colored_depth_registered_points" in lv:
            p = (RAW_DIR / val).resolve()
            if p.exists():
                pcd_path = p
        if lv.endswith(".txt") and "camera_color_info" in lv:
            p = (RAW_DIR / val).resolve()
            if p.exists():
                cam_path = p

    # fallbacks
    if pcd_path is None:
        for val in map(str, row.values):
            lv = val.lower()
            if lv.endswith(".pcd"):
                p = (RAW_DIR / val).resolve()
                if p.exists():
                    pcd_path = p
                    break

    if cam_path is None:
        for val in map(str, row.values):
            lv = val.lower()
            if lv.endswith(".txt") and "camera_color_info" in lv:
                p = (RAW_DIR / val).resolve()
                if p.exists():
                    cam_path = p
                    break

    return pcd_path, cam_path

def project_points(X, Y, Z, fx, fy, cx, cy, W, H):
    m = np.isfinite(Z) & (Z > 0)
    X, Y, Z = X[m], Y[m], Z[m]
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[keep], v[keep], Z[keep], keep

# ---------- PER IMAGE ----------
def process_image(img_name: str, grp: pd.DataFrame):
    rgb_path = RGB_DIR / img_name
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] RGB not found: {rgb_path}")
        return
    H, W = img.shape[:2]

    pcd_path, caminfo = pick_paths_from_pairs_colored(img_name)
    if pcd_path is None or caminfo is None:
        print(f"[WARN] colored PCD or color caminfo not found for {img_name}")
        return

    # --- project colored 3D points to the full image ---
    fx, fy, cx, cy = parse_caminfo_txt(caminfo)
    X, Y, Z, BGR = read_colored_pcd_ascii(pcd_path)
    u, v, _, keep = project_points(X, Y, Z, fx, fy, cx, cy, W, H)
    BGR = BGR[keep]

    # convert to integer pixel coords
    uu = u.astype(np.int32)
    vv = v.astype(np.int32)

    # optional global subsampling
    if uu.size > MAX_POINTS_ALL:
        idx_all = np.random.choice(uu.size, MAX_POINTS_ALL, replace=False)
        uu = uu[idx_all]
        vv = vv[idx_all]
        BGR = BGR[idx_all]

    # --- draw ALL projected points on the image ---
    for (px, py), (b, g, r) in zip(zip(uu, vv), BGR):
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(
                img,
                (px, py),
                DOT_RADIUS,
                (int(b), int(g), int(r)),
                DOT_THICKNESS,
            )

    # --- draw bboxes from labels (for visualization only) ---
    for _, r in grp.iterrows():
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)

        # shrink + optional trim
        x1 = max(0, x1 + BBOX_SHRINK_PX)
        y1 = max(0, y1 + BBOX_SHRINK_PX)
        x2 = min(W - 1, x2 - BBOX_SHRINK_PX)
        y2 = min(H - 1, y2 - BBOX_SHRINK_PX)
        if x2 <= x1 or y2 <= y1:
            continue

        if TRIM_TOP_FRAC > 0 or TRIM_BOTTOM_FRAC > 0:
            h = y2 - y1 + 1
            y1t = int(y1 + TRIM_TOP_FRAC * h)
            y2t = int(y2 - TRIM_BOTTOM_FRAC * h)
            if y2t > y1t:
                y1, y2 = y1t, y2t

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out = OUT_DIR / f"{Path(img_name).stem}_overlay.png"
    cv2.imwrite(str(out), img)
    print(f"[OK] {out}")

# ---------- DRIVER ----------
def main():
    ensure_dirs()
    boxes = load_boxes(IR_CSV)
    if boxes.empty:
        print("[INFO] No boxes to visualize.")
        return
    for img_name, grp in boxes.groupby("image"):
        process_image(str(img_name), grp)

if __name__ == "__main__":
    main()
