import os
from ultralytics import YOLO

print("COMPILING YOLOv8 TO TORCHSCRIPT")

# 1. Load a pre-trained YOLOv8 medium model
model = YOLO("yolov8m.pt")

# 2. Export the model architecture to TorchScript format
print("[INFO] Exporting static model graph...")
exported_path = model.export(format="torchscript", imgsz=640, device=0)

# 3. Move the output asset directly into the existing build folder
os.makedirs("build", exist_ok=True)
os.rename("yolov8m.torchscript", "build/yolov8m.torchscript")

print(f"[SUCCESS] Export complete! Model saved to: build/yolov8m.torchscript")
