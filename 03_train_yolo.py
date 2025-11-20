# train_yolo.py

import os, torch
from ultralytics import YOLO

os.environ.setdefault("WANDB_DISABLED", "true")

# Set high precision for float32 matrix multiplications if supported
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Path to the dataset YAML file
DATA = "FireExtinguisher.v2i.yolov11/data.yaml"

# Main training function
def main():
    
    model = YOLO("yolov8n.pt")  

    model.train(
        data=DATA,
        epochs=15,
        imgsz=384,         
        batch=8,           
        device="mps",
        workers=0,
        cache="disk",
        amp=True,
        patience=5,                             # early stop needs val=True
        val=True,                               # <— validate every epoch
        conf=0.30,          
        iou=0.60,
        max_det=50,
        plots=False,
        save_json=False,
        save_txt=False,
        project="runs",
        name="yolov8n_fire_extinguisher",
        seed=42,
        optimizer="auto",
    )

if __name__ == "__main__":
    main()
