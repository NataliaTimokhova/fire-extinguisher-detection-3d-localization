# 06_build_rgb_sensor_pairs.py
# Systematize multi-sensor files by pairing each RGB frame with the nearest
# files from every other stream. Saves a master CSV for downstream use.
#
# Output: raw/test/rgb_sensor_pairs.csv

from __future__ import annotations
from pathlib import Path
import re
import csv
import statistics

BASE = Path("raw/test")
OUT_CSV = BASE / "rgb_sensor_pairs.csv"

# Folders (adjust names here if your dataset uses different ones)
FOLDERS = {
    "rgb_img":   BASE / "camera_color_image_raw",
    "rgb_info":  BASE / "camera_color_camera_info",

    "depth_img": BASE / "camera_depth_image_raw",
    "depth_info":BASE / "camera_depth_camera_info",

    "pcd_raw":   BASE / "camera_depth_points",
    "pcd_color": BASE / "depth_registered_colored_pointclouds",

    "ir_left_img":  BASE / "camera_left_ir_image_raw",
    "ir_right_img": BASE / "camera_right_ir_image_raw",

    # Optional: if we have IR camera_info folders, add them here:
    "ir_left_info":  BASE / "camera_left_ir_camera_info",
    "ir_right_info": BASE / "camera_right_ir_camera_info",
}

NUM = re.compile(r"\d+")

def longest_num(s: str) -> int | None:
    """Return longest numeric chunk from a string as int (nanoseconds), else None."""
    parts = NUM.findall(s)
    if not parts:
        return None
    return int(max(parts, key=len))

def parse_camera_info_stamp(txt_path: Path) -> int | None:
    """
    Parse 'stamp: secs' and 'nsecs' from a camera_info .txt (ROS YAML-like).
    Returns nanoseconds since epoch, or None on failure.
    """
    try:
        secs = None
        nsecs = None
        for line in txt_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("secs:"):
                secs = int(line.split(":", 1)[1].strip())
            elif line.startswith("nsecs:"):
                nsecs = int(line.split(":", 1)[1].strip())
            if secs is not None and nsecs is not None:
                break
        if secs is None or nsecs is None:
            return None
        return secs * 1_000_000_000 + nsecs
    except Exception:
        return None

def scan_files_with_ts(folder: Path, exts: tuple[str, ...]) -> list[tuple[Path, int]]:
    """
    Return list of (file_path, ts_ns) for files with given extensions in folder.
    ts_ns comes from the longest numeric chunk in the filename.
    """
    out: list[tuple[Path, int]] = []
    if not folder or not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            ts = longest_num(p.stem)
            if ts is not None:
                out.append((p, ts))
    return out

def scan_info_files(folder: Path) -> list[tuple[Path, int]]:
    """
    Return list of (file_path, ts_ns) for camera_info .txt files.
    Prefer secs/nsecs from content; fall back to filename timestamp.
    """
    out: list[tuple[Path, int]] = []
    if not folder or not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() == ".txt":
            ts = parse_camera_info_stamp(p)
            if ts is None:
                ts = longest_num(p.stem)
            if ts is not None:
                out.append((p, ts))
    return out

def median_period_ms(pairs: list[tuple[Path, int]]) -> float | None:
    """Return median inter-frame period (ms) for a timestamped series."""
    if len(pairs) < 2:
        return None
    dts = []
    for i in range(1, len(pairs)):
        dts.append((pairs[i][1] - pairs[i-1][1]) / 1e6)  # ns → ms
    try:
        return float(statistics.median(dts))
    except statistics.StatisticsError:
        return None

def per_stream_tolerance_ms(series: list[tuple[Path, int]]) -> float:
    """
    Adaptive tolerance: tol_ms = max(5, 0.6 * median_period_ms).
    If period is unavailable, fall back to 300 ms.
    """
    period = median_period_ms(series)
    if period is None:
        return 300.0
    return max(5.0, 0.6 * period)

def nearest(ts_target: int, candidates: list[tuple[Path, int]]) -> tuple[Path | None, float | None]:
    """Pick candidate with minimal |dt|. Return (path, dt_ms). If no candidates, (None, None)."""
    best_path = None
    best_dt_ms = None
    for p, ts in candidates:
        dt_ms = abs(ts - ts_target) / 1e6
        if best_dt_ms is None or dt_ms < best_dt_ms:
            best_path = p
            best_dt_ms = dt_ms
    return best_path, best_dt_ms

def rel(p: Path | None) -> str:
    return str(p.relative_to(BASE)) if p else ""

def main():
    # --- Collect all series ---
    rgb_imgs   = scan_files_with_ts(FOLDERS["rgb_img"], (".png", ".jpg", ".jpeg"))
    rgb_infos  = scan_info_files(FOLDERS["rgb_info"])

    depth_imgs = scan_files_with_ts(FOLDERS["depth_img"], (".png",))
    depth_infos= scan_info_files(FOLDERS["depth_info"])

    pcd_raw    = scan_files_with_ts(FOLDERS["pcd_raw"], (".pcd",))
    pcd_color  = scan_files_with_ts(FOLDERS["pcd_color"], (".pcd",))

    ir_left    = scan_files_with_ts(FOLDERS["ir_left_img"], (".png",))
    ir_right   = scan_files_with_ts(FOLDERS["ir_right_img"], (".png",))

    # Optional IR camera_info (may be empty)
    ir_left_info  = scan_info_files(FOLDERS.get("ir_left_info", None))
    ir_right_info = scan_info_files(FOLDERS.get("ir_right_info", None))

    # --- Compute per-stream tolerances ---
    tols = {
        "rgb_info":  per_stream_tolerance_ms(rgb_infos),
        "depth_img": per_stream_tolerance_ms(depth_imgs),
        "depth_info":per_stream_tolerance_ms(depth_infos),
        "pcd_raw":   per_stream_tolerance_ms(pcd_raw),
        "pcd_color": per_stream_tolerance_ms(pcd_color),
        "ir_left":   per_stream_tolerance_ms(ir_left),
        "ir_right":  per_stream_tolerance_ms(ir_right),
        "ir_left_info":  per_stream_tolerance_ms(ir_left_info),
        "ir_right_info": per_stream_tolerance_ms(ir_right_info),
    }

    # --- Write CSV (one row per RGB frame) ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            # Master (RGB)
            "rgb_path","rgb_ts",

            # RGB info
            "rgb_info_path","rgb_info_dt_ms","rgb_info_within_tol","rgb_info_tol_ms",

            # Depth
            "depth_img_path","depth_img_dt_ms","depth_img_within_tol","depth_img_tol_ms",
            "depth_info_path","depth_info_dt_ms","depth_info_within_tol","depth_info_tol_ms",

            # Point clouds
            "depth_points_path","depth_points_dt_ms","depth_points_within_tol","depth_points_tol_ms",
            "depth_points_colored_path","depth_points_colored_dt_ms","depth_points_colored_within_tol","depth_points_colored_tol_ms",

            # IR
            "ir_left_path","ir_left_dt_ms","ir_left_within_tol","ir_left_tol_ms",
            "ir_right_path","ir_right_dt_ms","ir_right_within_tol","ir_right_tol_ms",

            # Optional IR camera_info
            "ir_left_info_path","ir_left_info_dt_ms","ir_left_info_within_tol","ir_left_info_tol_ms",
            "ir_right_info_path","ir_right_info_dt_ms","ir_right_info_within_tol","ir_right_info_tol_ms",
        ])

        for rgb_path, rgb_ts in rgb_imgs:
            # Nearest matches
            rgb_info_p,  rgb_info_dt  = nearest(rgb_ts, rgb_infos)
            depth_img_p, depth_img_dt = nearest(rgb_ts, depth_imgs)
            depth_info_p,depth_info_dt= nearest(rgb_ts, depth_infos)
            pcd_raw_p,   pcd_raw_dt   = nearest(rgb_ts, pcd_raw)
            pcd_col_p,   pcd_col_dt   = nearest(rgb_ts, pcd_color)
            irl_p,       irl_dt       = nearest(rgb_ts, ir_left)
            irr_p,       irr_dt       = nearest(rgb_ts, ir_right)
            irl_info_p,  irl_info_dt  = nearest(rgb_ts, ir_left_info)
            irr_info_p,  irr_info_dt  = nearest(rgb_ts, ir_right_info)

            # Within-tolerance flags per stream
            row = [
                rel(rgb_path), rgb_ts,

                rel(rgb_info_p), f"{rgb_info_dt:.3f}" if rgb_info_dt is not None else "", 
                (rgb_info_dt is not None and rgb_info_dt <= tols["rgb_info"]), f"{tols['rgb_info']:.3f}",

                rel(depth_img_p), f"{depth_img_dt:.3f}" if depth_img_dt is not None else "",
                (depth_img_dt is not None and depth_img_dt <= tols["depth_img"]), f"{tols['depth_img']:.3f}",

                rel(depth_info_p), f"{depth_info_dt:.3f}" if depth_info_dt is not None else "",
                (depth_info_dt is not None and depth_info_dt <= tols["depth_info"]), f"{tols['depth_info']:.3f}",

                rel(pcd_raw_p), f"{pcd_raw_dt:.3f}" if pcd_raw_dt is not None else "",
                (pcd_raw_dt is not None and pcd_raw_dt <= tols["pcd_raw"]), f"{tols['pcd_raw']:.3f}",

                rel(pcd_col_p), f"{pcd_col_dt:.3f}" if pcd_col_dt is not None else "",
                (pcd_col_dt is not None and pcd_col_dt <= tols["pcd_color"]), f"{tols['pcd_color']:.3f}",

                rel(irl_p), f"{irl_dt:.3f}" if irl_dt is not None else "",
                (irl_dt is not None and irl_dt <= tols["ir_left"]), f"{tols['ir_left']:.3f}",

                rel(irr_p), f"{irr_dt:.3f}" if irr_dt is not None else "",
                (irr_dt is not None and irr_dt <= tols["ir_right"]), f"{tols['ir_right']:.3f}",

                rel(irl_info_p), f"{irl_info_dt:.3f}" if irl_info_dt is not None else "",
                (irl_info_dt is not None and irl_info_dt <= tols["ir_left_info"]), f"{tols['ir_left_info']:.3f}",

                rel(irr_info_p), f"{irr_info_dt:.3f}" if irr_info_dt is not None else "",
                (irr_info_dt is not None and irr_info_dt <= tols["ir_right_info"]), f"{tols['ir_right_info']:.3f}",
            ]
            w.writerow(row)

    # Console summary (no files are changed or removed)
    print(f"Saved CSV: {OUT_CSV}")
    print("Per-stream tolerances (ms):")
    for k, v in tols.items():
        print(f"  {k:16s}: {v:.3f}")

if __name__ == "__main__":
    main()
