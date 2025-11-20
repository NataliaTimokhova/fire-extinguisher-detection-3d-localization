from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# -------- SETTINGS --------
BASE = Path("raw/test")
SENSORS = {
    "RGB": "camera_color_image_raw",
    "Depth": "camera_depth_image_raw",
    "IR Left": "camera_left_ir_image_raw",
    "IR Right": "camera_right_ir_image_raw",
    "Depth Points": "camera_depth_points",
    "Depth Colored": "depth_registered_colored_pointclouds",
}

# Plot options
MARKER_SIZE = 22
MARKER_ALPHA = 0.8
LINE_WIDTH = 0.7
JITTER_THRESHOLD_MS = 5.0
JITTER_MS = 1.0
X_MIN_MS = None
X_MAX_MS = None

NUM = re.compile(r"\d+")

def extract_timestamp(name: str):
    nums = NUM.findall(name)
    if not nums:
        return None
    return int(max(nums, key=len))

def analyze_folder(folder: Path) -> pd.DataFrame:
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    if not files:
        return pd.DataFrame(columns=["filename", "timestamp_ns"])

    rows = []
    for p in files:
        ts = extract_timestamp(p.stem)
        if ts is not None:
            rows.append({"filename": p.name, "timestamp_ns": ts})
    if not rows:
        return pd.DataFrame(columns=["filename", "timestamp_ns"])

    return pd.DataFrame(rows)

def build_combined(all_results: dict) -> pd.DataFrame:
    combined = []
    for sensor, df in all_results.items():
        for _, row in df.iterrows():
            combined.append({
                "sensor": sensor,
                "filename": row["filename"],
                "timestamp_ms": row["timestamp_ns"] / 1e6,
            })
    if not combined:
        return pd.DataFrame(columns=["sensor", "filename", "timestamp_ms"])
    return pd.DataFrame(combined).sort_values("timestamp_ms").reset_index(drop=True)

def visualize_timestamps(combined_df: pd.DataFrame):
    if combined_df.empty:
        print("[INFO] No data to plot.")
        return

    sensors = list(combined_df["sensor"].unique())
    t0 = combined_df["timestamp_ms"].min()

    plt.figure(figsize=(14, 6))

    for i, sensor in enumerate(sensors):
        sdf = combined_df[combined_df["sensor"] == sensor].copy().reset_index(drop=True)
        sdf["x"] = sdf["timestamp_ms"] - t0

        # minimal jitter for very-close consecutive frames within the same sensor
        for k in range(1, len(sdf)):
            dt_prev = sdf.loc[k, "timestamp_ms"] - sdf.loc[k-1, "timestamp_ms"]
            if dt_prev < JITTER_THRESHOLD_MS:
                jitter = JITTER_MS if (k % 2 == 0) else -JITTER_MS
                sdf.loc[k, "x"] += jitter

        plt.plot(sdf["x"], [i]*len(sdf), linewidth=LINE_WIDTH, alpha=0.6)
        plt.scatter(sdf["x"], [i]*len(sdf), s=MARKER_SIZE, alpha=MARKER_ALPHA, label=sensor)

    plt.yticks(range(len(sensors)), sensors)
    plt.xlabel("Time (ms since timestamp in dataset)")
    plt.title("Sensor Frame Timestamps — Files per Stream")
    plt.grid(True, alpha=0.3)
    if X_MIN_MS is not None or X_MAX_MS is not None:
        plt.xlim(left=X_MIN_MS, right=X_MAX_MS)
    plt.tight_layout()
    plt.legend(loc="upper right", frameon=True)
    plt.show()

def main():
    all_results = {}
    for name, subfolder in SENSORS.items():
        folder = BASE / subfolder
        if not folder.exists():
            print(f"[WARN] {name} folder not found: {folder}")
            continue
        df = analyze_folder(folder)
        all_results[name] = df

    combined_df = build_combined(all_results)
    print("\nCombined timestamp table (frames per sensor):")
    if not combined_df.empty:
        print(combined_df[["sensor", "filename", "timestamp_ms"]].to_string(index=False))
    else:
        print("[INFO] No timestamps extracted.")

    visualize_timestamps(combined_df)

if __name__ == "__main__":
    main()
