# ir_filter.py
# IR-based decoy filter using ONLY FFT entropy, and saving overlays for ALL RGB frames.
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# ---------- CONFIG ----------
BASE_DIR         = Path.cwd()
RAW_DIR          = BASE_DIR / "raw/test"
RGB_DIR          = RAW_DIR / "camera_color_image_raw"
PAIRS_CSV        = RAW_DIR / "rgb_sensor_pairs.csv"
DEPTH_FILTER_CSV = BASE_DIR / "runs/depth_filter/depth_filter_results.csv"

OUT_DIR     = BASE_DIR / "runs/ir_filter"
OUT_IMG_DIR = OUT_DIR / "images"
OUT_CSV     = OUT_DIR / "ir_filter.csv"

# Preprocessing to suppress the IR dot pattern
IR_GAUSS_BLUR_KSIZE = 7        # try 7 or 9
DOWNSCALE_FACTOR    = 0.5      # 0.4–0.6 works well
PERCENT_CLIP        = (1.0, 99.0)
MARGIN_SHRINK_PX    = 4        # shrink bbox before cropping IR

# Threshold (tune for your set)
T_FFT = 6.5                    # low ⇒ simple spectrum (flat poster/decoy)

# ---------- I/O HELPERS ----------
def ensure_dirs():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load boxes that were kept by depth filter
def load_kept_boxes(depth_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(depth_csv)
    df = df[df["decision"] == "real_kept"].copy()
    df["image"] = df["image"].apply(lambda s: Path(str(s)).name)
    for c in ["x1","y1","x2","y2"]:
        df[c] = df[c].astype(float).round().astype(int)
    return df

# From RGB-IR pairs, pick the best IR frame
def pick_ir_path(rows: pd.DataFrame):
    """Choose IR frame that is within tolerance and closest in time to the RGB."""
    candidates = []
    for side in ("left", "right"):
        tol = f"ir_{side}_within_tol"
        path = f"ir_{side}_path"
        dt   = f"ir_{side}_dt_ms"
        r = rows[rows.get(tol, False) == True]
        if r.empty:
            continue
        idx = r[dt].abs().astype(float).idxmin()
        rel = str(r.loc[idx, path]).strip()
        if not rel:
            continue
        abs_p = (RAW_DIR / rel).resolve()
        if abs_p.exists():
            candidates.append((abs(float(r.loc[idx, dt])), abs_p, side))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]

# Read IR image as grayscale uint8
def read_gray_u8(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img

# Match IR size to given (H,W)
def match_size(ir: np.ndarray, size_hw: tuple[int,int]) -> np.ndarray:
    H, W = size_hw
    if ir.shape[:2] == (H, W):
        return ir
    return cv2.resize(ir, (W, H), interpolation=cv2.INTER_NEAREST)

# Crop with margin shrink
def shrink_and_crop(img: np.ndarray, x1:int,y1:int,x2:int,y2:int, margin:int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1 = max(0, x1+margin); y1 = max(0, y1+margin)
    x2 = min(w-1, x2-margin); y2 = min(h-1, y2-margin)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return patch

# Percentile clipping to uint8
def percentile_clip_u8(patch: np.ndarray, low=1.0, high=99.0) -> np.ndarray:
    lo = np.percentile(patch, low)
    hi = np.percentile(patch, high)
    if hi <= lo:
        return patch.copy()
    p = np.clip(patch.astype(np.float32), lo, hi)
    p = (p - lo) / (hi - lo + 1e-6)
    return (p * 255.0).astype(np.uint8)

# ---------- METRICS ----------
def preprocess_patch(patch: np.ndarray) -> np.ndarray:
    p = percentile_clip_u8(patch, *PERCENT_CLIP)
    if IR_GAUSS_BLUR_KSIZE >= 3 and IR_GAUSS_BLUR_KSIZE % 2 == 1:
        p = cv2.GaussianBlur(p, (IR_GAUSS_BLUR_KSIZE, IR_GAUSS_BLUR_KSIZE), 0)
    if DOWNSCALE_FACTOR and DOWNSCALE_FACTOR < 1.0:
        p = cv2.resize(p, None, fx=DOWNSCALE_FACTOR, fy=DOWNSCALE_FACTOR, interpolation=cv2.INTER_AREA)
    return p

def fft_entropy_metric(patch_u8: np.ndarray, eps: float = 1e-9) -> float:
    f = np.fft.rfft2(patch_u8.astype(np.float32))
    mag = np.abs(f)
    mag = mag / (mag.sum() + eps)
    return float(-np.sum(mag * (np.log2(mag + eps))))

def decide(ent: float) -> str:
    return "ir_flat_decoy" if ent <= T_FFT else "ir_real_object"

# ---------- VIS ----------
def draw_overlay(rgb, x1,y1,x2,y2, ent, decision):
    color = (0,200,0) if decision == "ir_real_object" else (0,0,255)
    cv2.rectangle(rgb, (x1,y1), (x2,y2), color, 2)
    lines = [f"FFT:{ent:.2f} thr:{T_FFT}", decision]
    x, y = x1, max(12, y1-28)
    for i, t in enumerate(lines):
        cv2.putText(rgb, t, (x, y + i*12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

# ---------- MAIN ----------
def main():
    ensure_dirs()

    if not DEPTH_FILTER_CSV.exists():
        print(f"[ERROR] Missing {DEPTH_FILTER_CSV}")
        return
    if not PAIRS_CSV.exists():
        print(f"[ERROR] Missing {PAIRS_CSV}")
        return

    kept = load_kept_boxes(DEPTH_FILTER_CSV)
    pairs = pd.read_csv(PAIRS_CSV)

    out_rows = []
    processed_images = set()

    # Process images that have kept boxes from depth filter
    for img_name, group in kept.groupby("image"):
        processed_images.add(img_name)

        rgb_path = RGB_DIR / img_name
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            for _, r in group.iterrows():
                out_rows.append(dict(image=img_name, x1=r.x1,y1=r.y1,x2=r.x2,y2=r.y2,
                                     ir_side=None, ir_path=None,
                                     fft_entropy=None, decision="rgb_read_error"))
            continue

        pr = pairs[pairs["rgb_path"].str.endswith(img_name)]
        ir_path, side = pick_ir_path(pr)
        if ir_path is None:
            for _, r in group.iterrows():
                out_rows.append(dict(image=img_name, x1=r.x1,y1=r.y1,x2=r.x2,y2=r.y2,
                                     ir_side=None, ir_path=None,
                                     fft_entropy=None, decision="no_ir_available"))
            cv2.imwrite(str(OUT_IMG_DIR / img_name), rgb)
            continue

        ir = read_gray_u8(ir_path)
        if ir is None:
            for _, r in group.iterrows():
                out_rows.append(dict(image=img_name, x1=r.x1,y1=r.y1,x2=r.x2,y2=r.y2,
                                     ir_side=side, ir_path=str(ir_path),
                                     fft_entropy=None, decision="ir_read_error"))
            cv2.imwrite(str(OUT_IMG_DIR / img_name), rgb)
            continue

        ir = match_size(ir, rgb.shape[:2])
        vis = rgb.copy()

        for _, r in group.iterrows():
            x1,y1,x2,y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            patch = shrink_and_crop(ir, x1,y1,x2,y2, MARGIN_SHRINK_PX)
            if patch is None or patch.size < 25:
                out_rows.append(dict(image=img_name, x1=x1,y1=y1,x2=x2,y2=y2,
                                     ir_side=side, ir_path=str(ir_path),
                                     fft_entropy=None, decision="patch_invalid"))
                continue

            p = preprocess_patch(patch)
            ent = fft_entropy_metric(p)
            decision = decide(ent)

            out_rows.append(dict(
                image=img_name, x1=x1,y1=y1,x2=x2,y2=y2,
                ir_side=side, ir_path=str(ir_path),
                fft_entropy=round(ent, 3),
                decision=decision
            ))
            draw_overlay(vis, x1,y1,x2,y2, ent, decision)

        cv2.imwrite(str(OUT_IMG_DIR / img_name), vis)

    # Passthrough: save overlays (plain images) for RGB frames with NO kept boxes from depth filter
    all_rgb = sorted(p.name for p in RGB_DIR.glob("*.png"))
    leftover = [name for name in all_rgb if name not in processed_images]
    for img_name in leftover:
        rgb_path = RGB_DIR / img_name
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is not None:
            cv2.imwrite(str(OUT_IMG_DIR / img_name), rgb)
        # Add one CSV row to make it explicit this frame had no bboxes after depth
        out_rows.append(dict(
            image=img_name, x1=None, y1=None, x2=None, y2=None,
            ir_side=None, ir_path=None,
            fft_entropy=None,
            decision="no_boxes_from_depth"
        ))

    # Write CSV + small summary
    pd.DataFrame(out_rows).to_csv(OUT_CSV, index=False)
    kept_n = sum(1 for r in out_rows if r["decision"] == "ir_real_object")
    rej_n  = sum(1 for r in out_rows if r["decision"] == "ir_flat_decoy")
    no_ir  = sum(1 for r in out_rows if r["decision"] == "no_ir_available")
    nobox  = sum(1 for r in out_rows if r["decision"] == "no_boxes_from_depth")
    print(f"Done. Rows: {len(out_rows)} | kept:{kept_n} | rejected:{rej_n} | no IR:{no_ir} | no boxes from depth:{nobox}")
    print(f"CSV: {OUT_CSV}")
    print(f"Images: {OUT_IMG_DIR}")

if __name__ == "__main__":
    main()
