# data_analysis.py
import csv
import glob
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET_DIR = Path("FireExtinguisher.v2i.yolov11")
SPLITS = ["train", "valid", "test"]
OUTDIR = Path("data_analysis_output")


def list_images(split_dir: Path):
    """Return sorted list of image paths for a split."""
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend(glob.glob(str(split_dir / "images" / ext)))
    return sorted(imgs)


def read_labels(label_file: Path):
    """Return YOLO-normalized boxes: (cls, x, y, w, h)."""
    boxes = []
    try:
        with open(label_file, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 5:
                    cls, x, y, w, h = p
                    boxes.append(
                        (int(float(cls)), float(x), float(y), float(w), float(h))
                    )
    except FileNotFoundError:
        pass
    return boxes


def safe_hist(ax, data, bins=30, title="", xlabel="", xlim=None):
    """Draw a histogram or a 'no data' placeholder."""
    if not data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks([])
        if xlim:
            ax.set_xlim(*xlim)
        return
    ax.hist(data, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if xlim:
        ax.set_xlim(*xlim)


def plot_resolution_histograms(split_stats: dict, outdir: Path):
    """Resolution histograms as categorical bars."""
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) One figure per split with ALL resolutions (variable width to keep labels readable)
    for split in SPLITS:
        sizes = split_stats[split]["resolutions"]
        counts = Counter(sizes)
        labels_vals = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        labels, vals = zip(*labels_vals) if labels_vals else ([], [])

        # Save raw counts to CSV
        csv_path = outdir / f"image_sizes_{split}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["resolution", "count"])
            for k, v in labels_vals:
                w.writerow([k, v])


    # 2) Compact Top-15 summary across splits
    fig, axs = plt.subplots(1, 3, figsize=(14, 3), sharey=False)
    for i, split in enumerate(SPLITS):
        sizes = split_stats[split]["resolutions"]
        ax = axs[i]
        ax.set_title(split)
        ax.set_ylabel("count")
        if sizes:
            top = Counter(sizes).most_common(15)
            labs, cnts = zip(*top)
            x = np.arange(len(labs))
            ax.bar(x, cnts)
            ax.set_xticks(x)
            ax.set_xticklabels(labs, rotation=45, ha="right")
            ax.set_xlabel("resolution (WxH)")
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_xlabel("resolution (WxH)")
    fig.suptitle("Image sizes – Top 15 per split")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outdir / "image_size_hist_top15.png", dpi=150)
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Collect stats per split
    stats = {}
    for split in SPLITS:
        sd = DATASET_DIR / split
        images = list_images(sd) if sd.exists() else []
        labels_dir = sd / "labels"

        objs_per_img = []
        bbox_area_px = []
        resolutions = []

        for img_path in images:
            # Image size (resolution)
            try:
                with Image.open(img_path) as im:
                    W, H = im.width, im.height
            except Exception:
                W = H = None

            if W and H:
                resolutions.append(f"{W}x{H}")

            # Boxes
            boxes = read_labels(labels_dir / (Path(img_path).stem + ".txt"))
            objs_per_img.append(len(boxes))

            # Convert relative box sizes to pixel areas for this image
            if W and H and boxes:
                img_area = W * H
                for _, _, _, bw_rel, bh_rel in boxes:
                    bbox_area_px.append((bw_rel * bh_rel) * img_area)

        stats[split] = dict(
            images=images,
            objs_per_img=objs_per_img,
            bbox_area_px=bbox_area_px,
            resolutions=resolutions,
        )

        # Console summary
        print(f"\n[{split.upper()}]")
        print(f"images: {len(images)}")
        print(f"empty images (0 objects): {sum(1 for c in objs_per_img if c == 0)}")
        if bbox_area_px:
            print(
                f"object area px: mean={np.mean(bbox_area_px):.0f}, "
                f"min={np.min(bbox_area_px):.0f}, max={np.max(bbox_area_px):.0f}"
            )
        if resolutions:
            top = Counter(resolutions).most_common(5)
            txt = ", ".join([f"{r}×{c}" for r, c in top])
            print(f"top resolutions: {txt}")

    # 1) split_sizes.png — images per split (+ %)
    counts = [len(stats[s]["images"]) for s in SPLITS]
    total = sum(counts) or 1
    percents = [100 * c / total for c in counts]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    bars = ax.bar(SPLITS, counts)
    ax.set_ylabel("images")
    ax.set_title("Images per split")
    for b, p in zip(bars, percents):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{p:.1f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTDIR / "split_sizes.png", dpi=150)
    plt.close(fig)

    # 2) obj_per_image_hist.png — histograms (per split)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    for i, split in enumerate(SPLITS):
        data = stats[split]["objs_per_img"]
        maxc = max(data) if data else 0
        bins = np.arange(0, maxc + 2, 1)  # integer bins
        safe_hist(
            axs[i],
            data,
            bins=bins,
            title=split,
            xlabel="objects per image",
            xlim=(0, max(1, maxc)),
        )
    fig.suptitle("Objects per image")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / "obj_per_image_hist.png", dpi=150)
    plt.close(fig)

    # 3) obj_per_image_boxplot.png — one box per split
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.boxplot([stats[s]["objs_per_img"] for s in SPLITS], labels=SPLITS, showfliers=True)
    ax.set_title("Objects per image (boxplot)")
    ax.set_ylabel("objects per image")
    fig.tight_layout()
    fig.savefig(OUTDIR / "obj_per_image_boxplot.png", dpi=150)
    plt.close(fig)

    # 4) image_size_hist_{split}.png + image_size_hist_top15.png — categorical resolutions
    plot_resolution_histograms(stats, OUTDIR)

    # 5) obj_size_hist.png — histograms of object area (px)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3), sharey=False)
    for i, split in enumerate(SPLITS):
        data = stats[split]["bbox_area_px"]
        safe_hist(axs[i], data, bins=50, title=split, xlabel="object area (px)")
    fig.suptitle("Object area (pixels)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / "obj_size_hist.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved figures to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
