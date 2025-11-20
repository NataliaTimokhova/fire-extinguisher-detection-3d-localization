# 07_viz_custom.py
# Draw YOLO prediction bboxes on the custom images and save visualizations.
# Source images: raw/test/camera_color_image_raw
# Predictions:   runs/detect/custom88/labels
# Output:        runs/visualize/custom88/no_filter

import os
from pathlib import Path
import argparse
import cv2

# Accept both 5-col (cls x y w h) and 6-col (cls x y w h conf) YOLO TXT lines
def parse_yolo_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    x, y, w, h = map(float, parts[1:5])
    conf = float(parts[5]) if len(parts) >= 6 else None
    return cls, x, y, w, h, conf

def xywhn_to_xyxy(x, y, w, h, img_w, img_h):
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",   default="raw/test/camera_color_image_raw",
                    help="Folder with the 88 images (png/jpg/jpeg).")
    ap.add_argument("--pred",  default="runs/detect/custom88/labels",
                    help="Folder with YOLO TXT prediction files.")
    ap.add_argument("--out",   default="runs/visualize/custom88/no_filter",
                    help="Where to save images with drawn boxes.")
    ap.add_argument("--conf",  type=float, default=0.0,
                    help="Confidence threshold (keeps boxes with conf>=this when available).")
    ap.add_argument("--exts",  nargs="+", default=[".png", ".jpg", ".jpeg"],
                    help="Image extensions to include.")
    args = ap.parse_args()

    src_dir  = Path(args.src)
    pred_dir = Path(args.pred)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = set(e.lower() for e in args.exts)

    # Collect images
    images = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]
    images.sort()
    if not images:
        print(f"No images found in {src_dir} with extensions {sorted(exts)}")
        return

    drawn_count = 0
    missing_labels = 0
    empty_after_thresh = 0

    for img_path in images:
        stem = img_path.stem
        lbl_path = pred_dir / f"{stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: cannot read image {img_path}")
            continue
        H, W = img.shape[:2]

        if not lbl_path.exists():
            # No detections saved for this image
            missing_labels += 1
            # Still save a copy (optional). Comment the next line if need in blanks saved.
            cv2.imwrite(str(out_dir / img_path.name), img)
            continue

        kept = 0
        with open(lbl_path, "r") as f:
            for line in f:
                parsed = parse_yolo_line(line)
                if parsed is None:
                    continue
                cls, x, y, w, h, conf = parsed
                if conf is not None and conf < args.conf:
                    continue

                x1, y1, x2, y2 = xywhn_to_xyxy(x, y, w, h, W, H)

                # Thickness scales with image size a bit
                thick = max(1, int(round(0.002 * (W + H))))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick)

                label = f"extinguisher" if conf is None else f"extinguisher {conf:.2f}"
                t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, max(1, thick-1))
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                # draw filled label bg
                cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 4, y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                            max(1, thick - 1), cv2.LINE_AA)
                kept += 1

        if kept == 0:
            empty_after_thresh += 1

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        drawn_count += 1

    print(f"Processed images: {drawn_count}")
    print(f"Images with no label file: {missing_labels}")
    print(f"Images where all boxes were below conf<{args.conf}: {empty_after_thresh}")
    print(f"Saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
