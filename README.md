# Fire Extinguisher Detection, Decoy Filtering & 3D Localization

This repository contains the full pipeline for a Computer Vision project from the Luleå University of Technology
*Object Detection and Localization* course.  

The goal is to:

- Detect **real fire extinguishers** with YOLO,
- **Reject printed decoys** using depth and infrared (IR) information,
- Estimate the **3D distance** from the camera to each real extinguisher using point clouds.

The project is built and evaluated on:
- A labeled **Roboflow** dataset (YOLO training and validation),
- A multi-sensor **custom dataset** recorded from a real RGB–Depth–IR rig stored in `raw/test`.

---

## 1. Overall Pipeline

At a high level, the pipeline consists of the following stages:

1. **Roboflow dataset analysis and split**  
2. **YOLOv8-nano training** on the Roboflow dataset  
3. **YOLO inference** on the custom RGB dataset (88 images)  
4. **Custom dataset analysis and sensor synchronization**  
5. **Depth-based decoy filtering**  
6. **IR-based decoy filtering** using FFT entropy  
7. **3D point cloud projection and visualization**  
8. **3D distance estimation (localization)** inside the filtered bounding boxes  

The Python scripts in the repository implement these stages.

---

## 2. Directory Structure

```text
fire-extinguisher-detection-3d-localization/
│
├── 01_data_analysis.py                   # Roboflow dataset EDA
├── 02_data_split.py                      # Train/val/test split for Roboflow dataset
├── 03_train_yolo.py                      # YOLOv8-nano training script
├── 04_data_custom.py                     # Custom dataset folder/ID analysis
├── 05_analyze_timestamp_intervals.py     # Frame-rate and timestamp analysis per sensor
├── 06_build_rgb_sensor_pairs.py          # Build rgb_sensor_pairs.csv for multi-sensor sync
├── 07_viz_custom.py                      # Visualize YOLO predictions on custom RGB images
├── 08_depth_filter.py                    # Depth-based decoy filtering
├── 09_ir_filter.py                       # IR-based FFT-entropy filtering
├── 10_3d_points_bbox.py                  # Project depth points into RGB and bboxes
├── 11_3d_points_color_bbox.py            # Project colored point clouds into RGB
├── 12_3d_object_distance.py              # 3D distance (localization) estimation
│
├── Project_presentation.pdf
├── yolov8n.pt                            # Pretrained YOLOv8n checkpoint from Ultralytics
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 3. Scripts Overview and Roles

Below is a short description of each script and which part of the project it belongs to.

### 3.1 Roboflow Dataset and Model Training

- **`01_data_analysis.py`**  
  Exploratory Data Analysis (EDA) for the Roboflow dataset:
  - Counts images and labels,
  - Checks the distribution of objects per image,
  - Helps understand dataset size and variability.

- **`02_data_split.py`**  
  Creates a custom split of the Roboflow dataset:
  - Original: 90% train / 10% val (from Roboflow),
  - New split: 90% train / 6.7% val / 3.3% test,
  - Ensures enough training data while reserving a small test set for metric comparison.

- **`03_train_yolo.py`**  
  Trains the **YOLOv8-nano** model on the Roboflow dataset.  
  Key configuration:
  - Epochs: **15**
  - Image size: **384 × 384**
  - Batch size: **8**
  - Device: **MPS** (Apple GPU)
  - `cache="disk"`, `amp=True`, `patience=5`, `val=True`
  - Confidence threshold for evaluation: **0.30**
  - IoU threshold for NMS and mAP: **0.60**
  
  The final model and logs are stored in `runs/yolov8n_fire_extinguisher/`  
  (notably `weights/best.pt`).

### 3.2 Visualizing YOLO on the Custom Dataset

- **`07_viz_custom.py`**  
  Draws YOLO bounding boxes on the **custom RGB dataset** for visual inspection.
  It reads YOLO prediction text files and overlays them onto the original RGB images.

---

## 4. YOLO Inference on the Custom Dataset

After training, the YOLO model is run on the custom RGB images (`raw/test/camera_color_image_raw`) to obtain detections that will later be filtered by depth and IR.

Use the following command:

```bash
yolo predict   model="runs/yolov8n_fire_extinguisher/weights/best.pt"   source="raw/test/camera_color_image_raw"   imgsz=384 device=mps   conf=0.3  iou=0.5   save save_txt save_conf   project="runs/detect" name="custom88" exist_ok=True
```

This creates:

- Annotated RGB images with predicted bounding boxes,
- YOLO-format label files (`*.txt`) with class, bbox, and confidence,
- Outputs are stored under `runs/detect/custom88/` and later consumed by the filtering scripts.

---

## 5. Custom Dataset Analysis and Synchronization

The `raw/test` dataset is a fully unpacked `.bag` recording: each sensor stream (RGB, depth, IR, point clouds, camera info) is exported into its own folder. Sensors run at different frame rates, so synchronization is done by **timestamps in filenames**.

- **`04_data_custom.py`**  
  Inspects the structure of `raw/test`:
  - Counts files per sensor folder (RGB, depth, IR, point clouds),
  - Extracts numeric IDs (timestamps) from filenames,
  - Checks which frames and IDs are missing for each sensor.

- **`05_analyze_timestamp_intervals.py`**  
  Analyzes the time intervals between frames in each folder:
  - Converts long integer timestamps into milliseconds,
  - Computes median time deltas → effective frame rates per sensor,
  - Determines a tolerance window for synchronization between RGB and other sensors.

- **`06_build_rgb_sensor_pairs.py`**  
  Builds `rgb_sensor_pairs.csv`, which aligns each RGB image with:
  - The nearest depth image,
  - The nearest depth point cloud,
  - The nearest IR left/right images,
  - The nearest camera info files.

  This CSV is the central mapping used later by the depth, IR, and 3D localization scripts.

---

## 6. Depth-Based Decoy Filtering — `08_depth_filter.py`

Purpose: **remove flat posters/decoys using geometric depth information.**

Depth images are 16-bit maps where each pixel is a distance in millimetres. For each YOLO bounding box on an RGB frame:

1. The script finds the paired **depth image** from `rgb_sensor_pairs.csv`.
2. It crops the corresponding depth region inside the bbox.
3. It discards invalid depth pixels (`0` values).
4. It computes per-bbox metrics, in particular:
   - **Median depth** (typical distance of the object),
   - **Depth standard deviation** (how much depth varies within the box).

During analysis, **depth standard deviation** showed a clear pattern:

- Real extinguishers → high variation (the object has 3D thickness),
- Flat printed decoys → low variation (almost a plane).

A manually tuned threshold is used:

- `DEPTH_SD_THRESHOLD = 476.0`
  - `depth_std ≥ 476` → **real_kept** (likely real extinguisher),
  - `depth_std < 476` → **decoy_rejected**.

The script outputs:

- Filtered RGB images with **only kept boxes drawn in green**,  
- A CSV file `runs/depth_filter/depth_filter_results.csv` containing all bbox statistics and decisions.

Depth filter results (on the custom dataset):

- **TRR = 0.81** (slight decrease, model still meets TPR requirement),  
- **FPR = 0.21** (significant improvement from 0.79, but still above the target 0.15).

---

## 7. IR-Based Texture Filtering — `09_ir_filter.py`

Purpose: **further reject decoys using IR texture complexity (FFT entropy).**

This step operates only on boxes previously marked as `real_kept` by the depth filter.

For each such bbox:

1. The script picks the closest-in-time IR image (left or right) using `rgb_sensor_pairs.csv`.
2. The IR frame is resized to match RGB resolution.
3. The bbox is slightly shrunk to avoid borders and noise.
4. The IR patch inside the bbox is pre-processed:
   - Contrast normalization (1–99 percentile clipping),
   - Gaussian blur to suppress dot patterns,
   - Downscaling to speed up computations.

The **FFT-entropy** metric is then computed:

- FFT converts the patch into a frequency spectrum,
- Magnitudes are normalized to form a probability distribution,
- Shannon entropy of this distribution measures how complex the texture is.

Interpretation:

- **High FFT-entropy** → rich IR texture → **real object**,  
- **Low FFT-entropy** → flat/simple IR response → **decoy**.

A manually selected threshold is used:

- `T_FFT = 6.5`
  - `entropy > 6.5` → `ir_real_object` (kept),
  - `entropy ≤ 6.5` → `ir_flat_decoy` (rejected).

Outputs:

- `runs/ir_filter/images/` — RGB images with green boxes for real objects and red boxes for IR-decoys, with entropy values drawn,
- `runs/ir_filter/ir_filter.csv` (or manual version) — per-bbox entropy and decision.

IR filter results:

- **TRR = 0.81** (unchanged),  
- **FPR = 0.09** (now within the project requirement of ≤ 0.15).

---

## 8. 3D Point Projections — `10_3d_points_bbox.py` and `11_3d_points_color_bbox.py`

These scripts visualize how 3D geometry aligns with the RGB images.

- **`10_3d_points_bbox.py`**  
  Uses depth point clouds from `camera_depth_points`:
  - Loads depth point cloud and depth camera intrinsics,
  - Projects 3D points (X, Y, Z) into RGB pixel coordinates,
  - Shows points that fall **inside** each bbox, for debugging depth-data quality.

- **`11_3d_points_color_bbox.py`**  
  Uses colored registered point clouds from `depth_registered_colored_pointclouds`:
  - Reads 3D points + per-point RGB color,
  - Uses the RGB camera intrinsics to project them into the RGB frame,
  - Draws all projected points with true colors,
  - Overlays bounding boxes on top.

This helps visually verify correct sensor registration and camera calibration.

---

## 9. 3D Localization (Distance Estimation) — `12_3d_object_distance.py`

The final step estimates how far each remaining extinguisher is from the camera using depth point clouds.

For each RGB image with detections classified as `ir_real_object`:

1. Load the paired **depth point cloud** and **depth camera intrinsics** using `rgb_sensor_pairs.csv`.
2. Project all valid 3D points (X, Y, Z) into the RGB image.
3. For each bounding box:
   - Shrink it by **5 pixels** on all sides to avoid background,
   - Select all projected points inside the box,
   - From these, pick up to **5 points closest to the box center** (for robustness),
   - Compute the 3D range for each point:
     
     **range = sqrt(X² + Y² + Z²)**
     
   - Use the **median** of these ranges as the final distance estimate.

Using the median over a small set of central points makes the estimate robust against noise and stray background points.

Outputs:

- `runs/object_distance/images/` — RGB frames with:
  - Green bounding boxes,
  - Center dots,
  - Green distance labels (e.g. “2.34 m”),
- `runs/object_distance/object_distance.csv` — per-bbox distances, depth values and number of points used.

Some distances are biased due to dataset limitations (missing or sparse point clouds, large boxes whose center lies on background), but the method works well when good 3D coverage is available.

---

## 10. How to Run the Full Pipeline (Script Order)

The practical execution order, following the project structure, is:

1. **Roboflow dataset: analysis and split**
   ```bash
   python 01_data_analysis.py
   python 02_data_split.py
   ```

2. **Train YOLOv8-nano on Roboflow**
   ```bash
   python 03_train_yolo.py
   ```

3. **Run YOLO on the custom RGB dataset**
   ```bash
   yolo predict      model="runs/yolov8n_fire_extinguisher/weights/best.pt"      source="raw/test/camera_color_image_raw"      imgsz=384 device=mps      conf=0.3 iou=0.5      save save_txt save_conf      project="runs/detect" name="custom88" exist_ok=True
   ```

4. **(Optional) Inspect custom dataset structure**
   ```bash
   python 04_data_custom.py
   ```

5. **Analyze timestamps and build sensor pairing**
   ```bash
   python 05_analyze_timestamp_intervals.py
   python 06_build_rgb_sensor_pairs.py
   ```

6. **Depth-based decoy filter**
   ```bash
   python 08_depth_filter.py
   ```

7. **IR-based decoy filter**
   ```bash
   python 09_ir_filter.py
   ```

8. **3D visualization and distance estimation**
   ```bash
   python 10_3d_points_bbox.py
   python 11_3d_points_color_bbox.py
   python 12_3d_object_distance.py
   ```

---

## 11. Requirements and Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.  
The project was developed and tested on **macOS with Apple M4 GPU (MPS backend)**, but should work on other platforms with minor adjustments (e.g., `device="cpu"` or `device="cuda"`).

---

## 12. Notes and Limitations

- The custom dataset has **missing depth and point cloud frames**, and only **14 colored point clouds** for 88 RGB images. Some detections therefore cannot be localized in 3D.
- Sensors run at **different frame rates**, so synchronization via timestamps is approximate and based on tolerance windows.
- Large bounding boxes can have centers that fall on background, biasing the distance estimate upwards.

Despite these constraints, the pipeline:

- Meets the **True Positive Rate** requirement on the custom dataset,
- Reduces **False Positive Rate** from **0.79 → 0.09** after depth and IR filtering,
- Provides meaningful **3D distance estimates** for many of the real extinguishers.

This repository can serve as a template for multi-sensor object detection, decoy filtering,
and 3D localization in other applications.


## 13. License

### YOLOv8 (Ultralytics)
This project uses **YOLOv8n**, which is licensed under the **AGPL-3.0 License**.  
If you use YOLOv8 in your work, you must comply with its license:

**License Link:**  
https://github.com/ultralytics/ultralytics/blob/main/LICENSE

**Attribution Requirement:**  
```
Ultralytics YOLOv8 © 2023, AGPL-3.0 License
https://github.com/ultralytics/ultralytics
```

---

### Roboflow Dataset License
Training was performed using a public dataset hosted on Roboflow:

**Dataset:** Fire Extinguisher Detection  
**Link:**  
https://universe.roboflow.com/fire-extinguisher/fireextinguisher-z5atr/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

Please check the dataset page for license terms specific to that dataset.  
Most Roboflow datasets require attribution and restrict redistribution.

---

## Citations

### YOLOv8 Citation
If you use Ultralytics models in academic work:

```
@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ultralytics},
  title = {YOLOv8: Ultralytics},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
}
```

---

### Roboflow Dataset Citation
If you use the Fire Extinguisher dataset:

```
@misc{roboflow2023extinguisher,
  title = {Fire Extinguisher Dataset},
  howpublished = {Roboflow Universe},
  year = {2023},
  url = {https://universe.roboflow.com/fire-extinguisher/fireextinguisher-z5atr/},
}
```

---
