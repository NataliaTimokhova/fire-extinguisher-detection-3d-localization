# data_custom.py

"""
This script analyzes the dataset structure inside the folder raw/test and prints summary statistics.
It is primarily used during the initial stage of data inspection to determine:

- The number of files in each subfolder (e.g., camera_color_image_raw, camera_depth_image_raw, camera_depth_points, etc.);
- Whether filenames include timestamps (long numeric sequences);
- Whether IDs are aligned across different sensor folders, indicating potential synchronization;
- Which folders contain missing frames (for example, RGB = 88, depth = 48, IR = 76, etc.);
- Which files lack numeric IDs in their names.
"""


from pathlib import Path
import re
from collections import defaultdict

BASE = Path("raw/test")

# File extensions we’ll consider as “data files”
EXTS = {".png",".pcd", ".txt"}

# Folders to ignore (e.g., rosbags)
IGNORE_DIRS = {"rosbags", ".DS_Store", "__pycache__"}

ID_REGEX = re.compile(r"\d+")


def extract_numeric_id(stem: str):
    """
    Extract a numeric ID from a filename *stem*.
    - If multiple numeric chunks exist, we take the LONGEST one (typical for timestamps).
    - If no digits found, return None.
    """
    chunks = ID_REGEX.findall(stem)
    if not chunks:
        return None
    # choose the longest numeric chunk; if ties, the first longest
    return max(chunks, key=len)


def list_data_files(folder: Path):
    """Return list of files with known data extensions (non-recursive)."""
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            files.append(p)
    return sorted(files)


def main():
    if not BASE.exists():
        print(f"Base path not found: {BASE.resolve()}")
        return

    subdirs = [d for d in BASE.iterdir() if d.is_dir() and d.name not in IGNORE_DIRS]
    if not subdirs:
        print(f"No subfolders found under {BASE.resolve()}")
        return

    # Collect per-folder stats
    folder_counts = {}
    folder_ids = {}
    no_id_files = defaultdict(list)

    for d in sorted(subdirs, key=lambda x: x.name):
        files = list_data_files(d)
        folder_counts[d.name] = len(files)

        ids = set()
        for f in files:
            nid = extract_numeric_id(f.stem)
            if nid is None:
                no_id_files[d.name].append(f.name)
            else:
                ids.add(nid)
        folder_ids[d.name] = ids

    # Print counts
    print("\n=== File counts per folder ===")
    width = max(len(name) for name in folder_counts) if folder_counts else 10
    for name in sorted(folder_counts):
        print(f"{name:<{width}} : {folder_counts[name]}")

    # ID alignment (based on filename digits only)
    non_empty_id_sets = [s for s in folder_ids.values() if len(s) > 0]
    if not non_empty_id_sets:
        print("\nNo numeric IDs found in any filenames. Nothing to align.")
        # But still print which files lack IDs
        if any(no_id_files.values()):
            print("\nFiles without numeric IDs (by folder):")
            for k in sorted(no_id_files):
                if no_id_files[k]:
                    print(f"  {k}: {len(no_id_files[k])} files (e.g., {no_id_files[k][:3]})")
        return

    global_union = set().union(*non_empty_id_sets)
    global_intersection = set.intersection(*non_empty_id_sets) if len(non_empty_id_sets) > 1 else non_empty_id_sets[0]

    print("\n=== ID alignment summary (from filename digits) ===")
    print(f"Folders with at least one numeric ID: {', '.join(sorted([k for k,v in folder_ids.items() if v]))}")
    print(f"Total unique IDs across these folders (union): {len(global_union)}")
    print(f"IDs common to ALL such folders (intersection): {len(global_intersection)}")

    # Per-folder missing IDs
    print("\nMissing IDs per folder (union - folder_ids):")
    for name in sorted(folder_ids):
        ids = folder_ids[name]
        if not ids:
            print(f"  {name}: no numeric IDs found")
            continue
        missing = sorted(global_union - ids)
        print(f"  {name}: missing {len(missing)} IDs" + (f" (e.g., {missing[:5]})" if missing else ""))

    # Report files without any numeric ID in their name
    if any(no_id_files.values()):
        print("\nFiles without numeric IDs (by folder):")
        for k in sorted(no_id_files):
            if no_id_files[k]:
                examples = ", ".join(no_id_files[k][:5])
                print(f"  {k}: {len(no_id_files[k])} files (e.g., {examples})")

    print("\nDone.")


if __name__ == "__main__":
    main()
