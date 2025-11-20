"""
Overlay all projected 3D points (X,Y,Z) over the full RGB image.
Color each dot by its depth Z (global percentiles for contrast).
No filled patches, no gradients into the ROI — just points.
Bboxes from labels are drawn for reference only.

Inputs:
- raw/test/rgb_sensor_pairs.csv
- runs/ir_filter/ir_filter_manual.csv  (uses decision=='ir_real_object' if present)
- raw/test/camera_color_image_raw/*.png
- raw/test/camera_depth_points/depth_points_*.pcd   (ASCII, FIELDS x y z)
- raw/test/camera_depth_camera_info/camera_depth_info_*.txt (K line)

Output:
- runs/3d_points_bbox/<image_stem>_overlay.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# -------- PATHS --------
BASE_DIR = Path.cwd()
RAW_DIR  = BASE_DIR / "raw/test"
RGB_DIR  = RAW_DIR / "camera_color_image_raw"
PAIRS    = RAW_DIR / "rgb_sensor_pairs.csv"             # maps each RGB to its PCD + caminfo
IR_CSV   = BASE_DIR / "runs/ir_filter/ir_filter_manual.csv"

# (Optional, for your reference/documentation)
PCD_ROOT = RAW_DIR / "camera_depth_points"              # e.g., depth_points_*.pcd
CAM_ROOT = RAW_DIR / "camera_depth_camera_info"         # camera_depth_info_*.txt

OUT_DIR  = BASE_DIR / "runs/3d_points_bbox"

# -------- PARAMS (points only) --------
DOT_RADIUS       = 1          # small crisp dots
DOT_THICKNESS    = -1         # filled dots
MAX_POINTS_ALL   = 300000     # global subsample (avoid painting solid)
BBOX_SHRINK_PX   = 2          # avoid bbox border (for drawing only)
TRIM_TOP_FRAC    = 0.35       # used only when drawing box, if desired
TRIM_BOTTOM_FRAC = 0.35
CMAP             = cv2.COLORMAP_TURBO
FONT             = cv2.FONT_HERSHEY_SIMPLEX


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


def read_pcd_xyz(path: Path):
    fields = []
    lines  = []
    in_data = False
    mode = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for s in f:
            s = s.strip()
            if not in_data:
                if s.lower().startswith("fields "):
                    fields = s.split()[1:]
                elif s.lower().startswith("data "):
                    mode = s.split(None, 1)[1].lower()
                    in_data = True
                continue
            if s:
                lines.append(s)

    if mode != "ascii":
        raise ValueError("Only ASCII PCD supported")
    if not all(k in fields for k in ("x", "y", "z")):
        raise ValueError("PCD missing x y z")

    arr = np.loadtxt(lines, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]

    col = {n: i for i, n in enumerate(fields)}
    X = arr[:, col["x"]].astype(np.float32)
    Y = arr[:, col["y"]].astype(np.float32)
    Z = arr[:, col["z"]].astype(np.float32)
    return X, Y, Z


def pick_paths_from_pairs(img_name: str):
    df = pd.read_csv(PAIRS)
    if "rgb_path" not in df.columns:
        return None, None

    rows = df[df["rgb_path"].astype(str).str.endswith(img_name)]
    if rows.empty:
        return None, None

    row = rows.iloc[0]
    pcd_path = cam_path = None

    # Prefer depth PCD and depth caminfo, but fall back to any .pcd/.txt
    for val in map(str, row.values):
        lv = val.lower()
        if lv.endswith(".pcd") and "depth_points" in lv:
            p = (RAW_DIR / val).resolve()
            if p.exists():
                pcd_path = p
        if lv.endswith(".txt") and "camera_depth_info" in lv:
            p = (RAW_DIR / val).resolve()
            if p.exists():
                cam_path = p

    # fallbacks if not found by preferred names
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
            if lv.endswith(".txt") and "camera_depth_info" in lv:
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
    m2 = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[m2], v[m2], Z[m2]


def colors_from_z_global(z_vals: np.ndarray) -> np.ndarray:
    """1D depth -> (N,3) BGR using global 2–98% percentiles for contrast."""
    if z_vals.size == 0:
        return np.zeros((0, 3), np.uint8)
    zmin = float(np.percentile(z_vals, 2.0))
    zmax = float(np.percentile(z_vals, 98.0))
    if zmax <= zmin:
        zmax = zmin + 1e-6
    norm = np.clip((z_vals - zmin) / (zmax - zmin), 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(u8, CMAP).reshape(-1, 3)  # (N,3) BGR


def process_image(img_name: str, boxes: pd.DataFrame):
    rgb_path = RGB_DIR / img_name
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] RGB not found: {rgb_path}")
        return
    H, W = img.shape[:2]

    pcd_path, caminfo = pick_paths_from_pairs(img_name)
    if pcd_path is None or caminfo is None:
        print(f"[WARN] PCD/caminfo not found for {img_name}")
        return

    fx, fy, cx, cy = parse_caminfo_txt(caminfo)
    X, Y, Z = read_pcd_xyz(pcd_path)
    u, v, z = project_points(X, Y, Z, fx, fy, cx, cy, W, H)

    # Global subsampling to avoid over-drawing
    if u.size > MAX_POINTS_ALL:
        idx_all = np.random.choice(u.size, MAX_POINTS_ALL, replace=False)
        u = u[idx_all]
        v = v[idx_all]
        z = z[idx_all]

    # Compute depth-based colors globally
    cols = colors_from_z_global(z)  # (N,3) BGR

    uu = u.astype(np.int32)
    vv = v.astype(np.int32)

    # ---- draw ALL points on the image (full image, no bbox filtering) ----
    for (px, py), (b, g, r) in zip(zip(uu, vv), cols):
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(
                img,
                (px, py),
                DOT_RADIUS,
                (int(b), int(g), int(r)),
                DOT_THICKNESS,
            )

    # ---- draw bboxes (optionally trimmed) for visualization only ----
    for _, r in boxes.iterrows():
        x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)

        # shrink bbox to avoid border
        x1 = max(0, x1 + BBOX_SHRINK_PX)
        y1 = max(0, y1 + BBOX_SHRINK_PX)
        x2 = min(W - 1, x2 - BBOX_SHRINK_PX)
        y2 = min(H - 1, y2 - BBOX_SHRINK_PX)
        if x2 <= x1 or y2 <= y1:
            continue

        # trim top/bottom if desired (only affects box, not which points are drawn)
        h = y2 - y1 + 1
        y1t = int(y1 + TRIM_TOP_FRAC * h)
        y2t = int(y2 - TRIM_BOTTOM_FRAC * h)
        if y2t <= y1t:
            y1t, y2t = y1, y2

        cv2.rectangle(img, (x1, y1t), (x2, y2t), (0, 255, 0), 2)

    out = OUT_DIR / f"{Path(img_name).stem}_overlay.png"
    cv2.imwrite(str(out), img)
    print(f"[OK] {out}")


def main():
    ensure_dirs()
    df = load_boxes(IR_CSV)
    if df.empty:
        print("[INFO] No boxes.")
        return
    for img_name, grp in df.groupby("image"):
        process_image(str(img_name), grp)


if __name__ == "__main__":
    main()
