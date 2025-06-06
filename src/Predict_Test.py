from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("Model/YoloPrediction/my_model.pt")

# Path to the test image
img_path = "Data/train_10.png"

# Run prediction
results = model(img_path)

# Display the result with the detections
for r in results:
    r.show()  