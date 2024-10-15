#flask_api\model.py

import torch
import torch.nn as nn
import os
import pickle
import numpy as np
from torchvision import transforms, models
from transformers import DeiTForImageClassification, CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load SimCLR Model
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLR, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x).squeeze()
        z = self.projection_head(h)
        return h, z

def load_model(model_path, model):
    # Corrected: No 'weights_only' argument in load_state_dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()  # Set to evaluation mode
    return model

# Load models
def load_models():
    simclr_model_instance = SimCLR(models.resnet50(weights='ResNet50_Weights.DEFAULT'), out_dim=128)
    simclr_model = load_model('models/simclr_model1.pth', simclr_model_instance)

    deit_model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224', num_labels=128)
    deit_model.load_state_dict(torch.load('models/deit_finetuned_improved.pth', map_location=torch.device('cpu')))

    clip_model = load_model('models/clip_finetuned1.pth', CLIPModel.from_pretrained("openai/clip-vit-base-patch32"))
    
    base_cnn_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')  # Load default ResNet18
    base_cnn_model.eval()  # Set to evaluation mode

    return simclr_model, deit_model, clip_model, base_cnn_model

# Image transform for non-CLIP models
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Feature extraction functions
def extract_features(model, image, model_type):
    with torch.no_grad():
        image_tensor = get_image_transform()(image).unsqueeze(0)
        if model_type == 'simclr':
            return model(image_tensor)[1]
        elif model_type == 'deit':
            inputs = {"pixel_values": image_tensor}
            outputs = model(**inputs)
            return outputs.logits
        elif model_type == 'clip':
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_inputs = processor(images=image, return_tensors="pt")
            image_features = model.get_image_features(**clip_inputs)
            return image_features.cpu().numpy().reshape(1, -1)
        elif model_type == 'base_cnn':
            return model(image_tensor).detach().cpu().numpy()

# Load and save features
def load_features(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded features from {filename}.")
        return features
    return {}

def save_features_to_file(features, filename):
    with open(filename, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {filename}.")

# Find most similar images
def find_most_similar_image(uploaded_features, dataset_features):
    uploaded_features = uploaded_features.reshape(1, -1)  # Ensure shape is [1, feature_dim]
    dataset_features_array = np.array(list(dataset_features.values()))
    dataset_features_array = dataset_features_array.reshape(dataset_features_array.shape[0], -1)

    similarities_cosine = cosine_similarity(uploaded_features, dataset_features_array)

    most_similar = sorted(zip(dataset_features.keys(), similarities_cosine[0]),
                          key=lambda item: item[1], reverse=True)

    return most_similar

# Load dataset features
def load_dataset_features(image_folder, model, model_type, feature_file):
    features = load_features(feature_file)
    if features:
        return features  # Skip extraction if features are already loaded

    features = {}
    img_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))]

    print(f"Loading features for {len(img_files)} images...")
    
    for img_file in img_files:
        img_path = os.path.join(image_folder, img_file)
        image = Image.open(img_path).convert("RGB")
        feature = extract_features(model, image, model_type)
        features[img_path] = feature

    save_features_to_file(features, feature_file)
    return features
