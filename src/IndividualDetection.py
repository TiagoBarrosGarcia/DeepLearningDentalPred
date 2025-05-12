from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load model
model = YOLO("Model/YoloPrediction/my_model.pt")

# Path to input image
img_path = "Data/train_10.png"
img = cv2.imread(img_path)

# Run prediction
results = model(img_path)
boxes = results[0].boxes
class_names = model.names

# Output folder
output_dir = "Data/individual_detection"
os.makedirs(output_dir, exist_ok=True)

# For each box, isolate it and let YOLO render only that box
for i, box in enumerate(boxes):
    single_result = results[0]  
    single_result = single_result.new()  
    single_result.orig_img = img
    single_result.boxes = boxes[i:i+1] 

    rendered_img = single_result.plot()

    class_id = int(box.cls[0])
    class_name = class_names[class_id]
    filename = f"box_{i}_{class_name}.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), rendered_img)

print(f"Saved {len(boxes)} images in '{output_dir}' — each with one box rendered by YOLO.")
