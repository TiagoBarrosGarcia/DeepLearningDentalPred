from ultralytics import YOLO
import cv2
import os
from datetime import datetime
from ultralytics.engine.results import Results

# Load the model once globally to avoid reloading
model = YOLO("Model/modelo_treinado-2.pth")

from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load the model once globally to avoid reloading
model = YOLO("Model/YoloPrediction/my_model.pt")

def save_bboxes_from_image(img_path, output_dir="Data/individual_detection", yolo_output_dir="Data/yolo_results"):
    import os
    import cv2
    from datetime import datetime

    COLOR_MAP = {
        'Caries': (0, 0, 255),            # Red
        'Deep Caries': (128, 0, 128),     # Purple
        'Impacted': (255, 0, 0),          # Blue
        'Periapical Lesion': (0, 255, 0)  # Green
    }

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(yolo_output_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

    results = model(img_path)[0]
    boxes = results.boxes
    class_names = model.names

    bbox_info = []

    for i in range(len(boxes)):
        img_copy = img.copy()

        box = boxes[i]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        class_id = int(box.cls[0])
        class_name = class_names[class_id]

        color = COLOR_MAP.get(class_name, (255, 255, 255))  # Branco como fallback
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 4)
        cv2.putText(img_copy, f"ID: {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        filename = f"yolo_{i}_{class_name}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, img_copy)

        bbox_info.append({
            'id': i,
            'filename': filename,
            'class_name': class_name,
            'bbox': [x1, y1, x2, y2],
            'color': color
        })

    # Imagem com todas as bboxes
    full_img = img.copy()
    for i in range(len(boxes)):
        box = boxes[i]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        color = COLOR_MAP.get(class_name, (255, 255, 255))
        cv2.rectangle(full_img, (x1, y1), (x2, y2), color, 4)
        cv2.putText(full_img, f"ID: {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    yolo_full_path = os.path.join(yolo_output_dir, f"{base_filename}_{timestamp}.png")
    cv2.imwrite(yolo_full_path, full_img)

    print(f"Imagem com todas as bboxes salva: {yolo_full_path}")
    print(f"{len(boxes)} imagens salvas com apenas uma bbox desenhada cada.")

    return bbox_info, yolo_full_path, COLOR_MAP


image_path = 'Data/train_8.png'
save_bboxes_from_image(image_path)
