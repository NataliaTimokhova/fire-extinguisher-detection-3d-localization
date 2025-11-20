# depth_filter.py
# Use existing label files only. Pair RGB<->Depth via rgb_sensor_pairs.csv
# (which stores paths relative to raw/test/). Keep bboxes whose depth std-dev
# is high (non-flat). Save EVERY RGB frame; draw only surviving boxes (green).
# Write results to runs/depth_filter/depth_filter_results.csv.

from pathlib import Path
import csv
import sys
import numpy as np
import pandas as pd
import cv2

# ---------------- CONFIG ----------------
ROOT       = Path.cwd()
DATA_BASE  = ROOT / "raw/test"  # IMPORTANT: CSV paths are relative to this dir
RGB_DIR    = DATA_BASE / "camera_color_image_raw"
LABEL_DIR  = ROOT / "runs/detect/custom88/labels"
PAIR_CSV   = DATA_BASE / "rgb_sensor_pairs.csv"
OUTPUT_DIR = ROOT / "runs/depth_filter"

# keep bbox if depth std-dev (native units; uint16 depth is usually millimeters) >= threshold
DEPTH_SD_THRESHOLD = 476.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utilities ----------------
def build_pair_indices(csv_path: Path):
    """
    Read rgb_sensor_pairs.csv and build two indices:
      - by absolute RGB path (under DATA_BASE)
      - by RGB basename (fallback)
    Returns (abs_map, name_map) where values are absolute depth paths (Path or None).
    """
    abs_map = {}
    name_map = {}
    if not csv_path.exists():
        print(f"[ERROR] Pair CSV not found: {csv_path}", file=sys.stderr)
        return abs_map, name_map

    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rgb_rel   = (row.get("rgb_path", "") or "").strip()
            depth_rel = (row.get("depth_img_path", "") or "").strip()
            if not rgb_rel:
                continue

            rgb_abs   = (DATA_BASE / rgb_rel).resolve()
            depth_abs = (DATA_BASE / depth_rel).resolve() if depth_rel else None

            abs_map[str(rgb_abs)] = depth_abs
            name_map[Path(rgb_rel).name] = depth_abs
    return abs_map, name_map

def yolo_txt_to_bboxes(txt_path: Path, img_w: int, img_h: int):
    """Read YOLO .txt -> [{'cls': int, 'xyxy': (x1,y1,x2,y2)}]."""
    boxes = []
    if not txt_path.exists():
        return boxes
    with txt_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w  = float(parts[3]) * img_w
            h  = float(parts[4]) * img_h
            x1 = int(round(cx - w / 2));  y1 = int(round(cy - h / 2))
            x2 = int(round(cx + w / 2));  y2 = int(round(cy + h / 2))
            x1 = max(0, min(img_w - 1, x1)); y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w - 1, x2)); y2 = max(0, min(img_h - 1, y2))
            if x2 > x1 and y2 > y1:
                boxes.append({"cls": cls, "xyxy": (x1, y1, x2, y2)})
    return boxes

def load_depth(path: Path):
    """Load depth image preserving bit depth."""
    if not path or not path.exists():
        return None
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"[WARN] Failed to read depth: {path}", file=sys.stderr)
    return depth

def ensure_same_size(depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Resize depth to RGB size if needed (nearest-neighbor)."""
    if depth is None or rgb is None:
        return depth
    if depth.shape[:2] != rgb.shape[:2]:
        return cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return depth

# ---------------- Main ----------------
def main():
    abs_map, name_map = build_pair_indices(PAIR_CSV)
    if not abs_map and not name_map:
        print("[ERROR] Could not build depth mapping from CSV.", file=sys.stderr)
        return

    rgb_list = sorted(list(RGB_DIR.glob("*.png")) + list(RGB_DIR.glob("*.jpg")))
    if not rgb_list:
        print(f"No RGB images in {RGB_DIR}")
        return

    rows = []
    saved = 0
    total_boxes = 0
    kept_boxes = 0

    for rgb_path in rgb_list:
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            cv2.imwrite(str(OUTPUT_DIR / rgb_path.name), np.zeros((10,10,3), np.uint8))
            rows.append({"image": rgb_path.name, "class_id": None, "x1": None, "y1": None,
                         "x2": None, "y2": None, "depth_median_native": None,
                         "depth_std_native": None, "decision": "rgb_read_error"})
            saved += 1
            continue

        H, W = rgb.shape[:2]

        # Depth path: prefer absolute-key match; fallback to basename match
        depth_path = abs_map.get(str(rgb_path.resolve()), None)
        if depth_path is None:
            depth_path = name_map.get(rgb_path.name, None)

        depth = load_depth(depth_path) if depth_path else None
        if depth is not None:
            depth = ensure_same_size(depth, rgb)

        lbl_path = (LABEL_DIR / (rgb_path.stem + ".txt"))
        boxes = yolo_txt_to_bboxes(lbl_path, W, H) if lbl_path.exists() else []

        kept_for_draw = []
        any_label = bool(boxes)

        if boxes and depth is not None:
            for b in boxes:
                x1, y1, x2, y2 = b["xyxy"]
                total_boxes += 1
                crop = depth[y1:y2, x1:x2]
                valid = crop[crop > 0]
                if valid.size == 0:
                    rows.append({"image": rgb_path.name, "class_id": b["cls"],
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "depth_median_native": None,
                                 "depth_std_native": None,
                                 "decision": "no_depth_data"})
                    continue

                d_med = float(np.median(valid))
                d_std = float(np.std(valid))
                decision = "real_kept" if d_std >= DEPTH_SD_THRESHOLD else "decoy_rejected"

                rows.append({"image": rgb_path.name, "class_id": int(b['cls']),
                             "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "depth_median_native": round(d_med, 2),
                             "depth_std_native": round(d_std, 2),
                             "decision": decision})
                if decision == "real_kept":
                    kept_for_draw.append((x1, y1, x2, y2))
                    kept_boxes += 1

        elif boxes and depth is None:
            for b in boxes:
                total_boxes += 1
                rows.append({"image": rgb_path.name, "class_id": int(b['cls']),
                             "x1": b["xyxy"][0], "y1": b["xyxy"][1],
                             "x2": b["xyxy"][2], "y2": b["xyxy"][3],
                             "depth_median_native": None,
                             "depth_std_native": None,
                             "decision": "no_depth_available"})
        else:
            rows.append({"image": rgb_path.name, "class_id": None,
                         "x1": None, "y1": None, "x2": None, "y2": None,
                         "depth_median_native": None, "depth_std_native": None,
                         "decision": "no_labels"})

        vis = rgb.copy()
        for (x1, y1, x2, y2) in kept_for_draw:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(OUTPUT_DIR / rgb_path.name), vis)
        saved += 1

    out_csv = OUTPUT_DIR / "depth_filter_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved frames: {saved}")
    print(f"Total boxes (label files): {total_boxes}")
    print(f"Boxes kept after depth-std filter: {kept_boxes}")
    print(f"Results CSV: {out_csv}")
    print(f"Images written to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
