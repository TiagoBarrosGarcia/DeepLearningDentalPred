import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18
from PIL import Image
import os

def load_resnet_model(model_path, train_data_dir):
    """
    Loads a trained ResNet model for classification, adapting if class count differs.

    Args:
        model_path (str): Path to the trained model weights (.pth file).
        train_data_dir (str): Directory used for training images to infer class names.

    Returns:
        model (torch.nn.Module): Loaded ResNet model in eval mode.
        classes (list): List of class names.
        device (torch.device): Device on which the model is loaded.
        transform (torchvision.transforms.Compose): Image transformations to apply.
    """
    classes = datasets.ImageFolder(root=train_data_dir).classes
    num_classes = len(classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load state_dict with possible mismatch in fc layer
    state_dict = torch.load(model_path, map_location=device)

    # Verificar se o número de classes coincide com o modelo salvo
    saved_fc_weight = state_dict.get('fc.weight')
    if saved_fc_weight is not None and saved_fc_weight.shape[0] != num_classes:
        print(f"[INFO] Mismatch de classes detectado: modelo salvo tem {saved_fc_weight.shape[0]} classes, atual tem {num_classes}")
        # Remover fc weights
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return model, classes, device, transform


def predict_folder(model, classes, device, transform, image_folder):
    """
    Predict classes for all images in a folder.

    Args:
        model (torch.nn.Module): Loaded classification model.
        classes (list): List of class names.
        device (torch.device): Device for computation.
        transform (torchvision.transforms.Compose): Transformations to apply to input images.
        image_folder (str): Path to the folder with images to classify.

    Returns:
        dict: {filename (str): predicted_class (str)}
    """
    predictions = {}

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = output.argmax(dim=1).item()
            predicted_class = classes[pred]
            predictions[filename] = predicted_class
            print(f'Image: {filename} -> Predicted class: {predicted_class}')

    return predictions
