import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import DeiTForImageClassification, CLIPModel, CLIPProcessor
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle

# Load SimCLR Model
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLR, self).__init__()
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # Remove the classification layer
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x).squeeze()  # Get the backbone representation
        z = self.projection_head(h)  # Get the projection
        return h, z

def load_simclr_model(model_path='simclr_model.pth'):
    base_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    simclr_model = SimCLR(base_model, out_dim=128)
    simclr_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    simclr_model.eval()
    return simclr_model

# Load DeiT Model
def load_deit_model(model_path='deit_finetuned.pth'):
    deit_model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224', num_labels=128)
    deit_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    deit_model.eval()
    return deit_model

# Load CLIP Model
def load_clip_model(model_path='clip_finetuned.pth'):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Load Base CNN Model
def load_base_cnn_model():
    base_cnn_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')  # Load pre-trained ResNet18 model
    base_cnn_model.eval()  # Set to evaluation mode
    return base_cnn_model

# Image transform for models
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Extract features using SimCLR and DeiT
def extract_features(simclr_model, deit_model, image):
    with torch.no_grad():
        image_tensor = get_image_transform()(image).unsqueeze(0)
        simclr_features = simclr_model(image_tensor)[1]
        deit_features = deit_model(image_tensor).logits
    return simclr_features, deit_features

# Extract features using CLIP
def extract_clip_features(clip_model, image):
    with torch.no_grad():
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_inputs = clip_processor(images=image, return_tensors="pt")
        clip_features = clip_model.get_image_features(**clip_inputs)
    return clip_features

# Extract features using Base CNN
def extract_base_cnn_features(base_cnn_model, image):
    with torch.no_grad():
        image_tensor = get_image_transform()(image).unsqueeze(0)
        base_cnn_features = base_cnn_model(image_tensor).detach().cpu().numpy()
    return base_cnn_features

# Save features to a pickle file
def save_features_to_file(features, filename='features.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {filename}.")

# Load pre-extracted features for dataset images (SimCLR + DeiT)
def load_dataset_features(image_folder, feature_file='features.pkl'):
    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded features from {feature_file}.")
        return features

    features = {}
    total_images = len(os.listdir(image_folder))
    print(f"Loading features for {total_images} images...")
    start_time = time.time()

    for img_file in os.listdir(image_folder):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(image_folder, img_file)
            image = Image.open(img_path).convert("RGB")
            simclr_feature, deit_feature = extract_features(simclr_model, deit_model, image)
            features[img_path] = (simclr_feature.numpy(), deit_feature.numpy())
            if len(features) % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {len(features)} images in {elapsed_time:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"Finished loading features for {total_images} images in {total_time:.2f} seconds.")

    save_features_to_file(features)
    return features

# Load dataset features for CLIP
def load_clip_dataset_features(image_folder, clip_model, batch_size=32, feature_file='clip_features.pkl'):
    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded features from {feature_file}.")
        return features

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = clip_model.to(device)

    features = {}
    image_paths = [os.path.join(image_folder, img_file) for img_file in os.listdir(image_folder) if img_file.endswith(('.jpg', '.png'))]
    total_images = len(image_paths)
    print(f"Loading features for {total_images} images...")

    start_time = time.time()
    batch_images = []

    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        batch_images.append(image)

        if len(batch_images) == batch_size or i == total_images - 1:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                batch_features = clip_model.get_image_features(**inputs)
            
            batch_features = batch_features.cpu().numpy()
            for             j, image_path in enumerate(batch_images):
                features[image_paths[i - len(batch_images) + 1 + j]] = batch_features[j]

            batch_images = []

            if (i + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {i + 1}/{total_images} images in {elapsed_time:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"Finished loading features for {total_images} images in {total_time:.2f} seconds.")

    with open(feature_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {feature_file}.")

    return features

# Load dataset features for Base CNN
def load_base_cnn_dataset_features(image_folder, base_cnn_model, batch_size=32, feature_file='base_cnn_features.pkl'):
    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
        print(f"Loaded features from {feature_file}.")
        return features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_cnn_model = base_cnn_model.to(device)

    features = {}
    image_paths = [os.path.join(image_folder, img_file) for img_file in os.listdir(image_folder) if img_file.endswith(('.jpg', '.png'))]
    total_images = len(image_paths)
    print(f"Loading features for {total_images} images...")

    start_time = time.time()
    batch_images = []

    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        batch_images.append(image)

        if len(batch_images) == batch_size or i == total_images - 1:
            batch_images_tensor = torch.stack([get_image_transform()(img) for img in batch_images]).to(device)
            with torch.no_grad():
                batch_features = base_cnn_model(batch_images_tensor).cpu().numpy()
            
            for j, img_path in enumerate(batch_images):
                features[image_paths[i - len(batch_images) + 1 + j]] = batch_features[j]

            batch_images = []

            if (i + 1) % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {i + 1}/{total_images} images in {elapsed_time:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"Finished loading features for {total_images} images in {total_time:.2f} seconds.")

    with open(feature_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {feature_file}.")

    return features

# Find the most similar image using features from SimCLR, DeiT, and optionally CLIP or Base CNN
def find_most_similar_image(uploaded_features, dataset_features, clip_features=None, base_cnn_features=None, method='all'):
    similarities = {}
    
    uploaded_simclr_feature, uploaded_deit_feature = uploaded_features
    uploaded_simclr_feature = uploaded_simclr_feature.reshape(1, -1)
    uploaded_deit_feature = uploaded_deit_feature.reshape(1, -1)

    for img_path, (feature_simclr, feature_deit) in dataset_features.items():
        feature_simclr = feature_simclr.reshape(1, -1)
        feature_deit = feature_deit.reshape(1, -1)

        # Compute cosine similarity for SimCLR and DeiT
        simclr_similarity = cosine_similarity(uploaded_simclr_feature, feature_simclr)
        deit_similarity = cosine_similarity(uploaded_deit_feature, feature_deit)

        if method == 'simclr':
            avg_similarity = simclr_similarity
        elif method == 'deit':
            avg_similarity = deit_similarity
        elif method == 'clip' and clip_features is not None:
            clip_feature = clip_features[img_path].reshape(1, -1)
            clip_similarity = cosine_similarity(uploaded_clip_feature, clip_feature)
            avg_similarity = (simclr_similarity + deit_similarity + clip_similarity) / 3
        elif method == 'base_cnn' and base_cnn_features is not None:
            base_cnn_feature = base_cnn_features[img_path].reshape(1, -1)
            base_cnn_similarity = cosine_similarity(uploaded_base_cnn_feature, base_cnn_feature)
            avg_similarity = base_cnn_similarity
        else:  # all
            avg_similarity = (simclr_similarity + deit_similarity + clip_similarity) / 3 if clip_features else (simclr_similarity + deit_similarity) / 2

        similarities[img_path] = avg_similarity.item()

    most_similar = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return most_similar

# Streamlit application
st.title("Image Matching Application")

# Load models
simclr_model = load_simclr_model()
deit_model = load_deit_model()
clip_model = load_clip_model()
base_cnn_model = load_base_cnn_model()

# Upload an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Extract features from the uploaded image using SimCLR and DeiT
    uploaded_features = extract_features(simclr_model, deit_model, uploaded_image)

    # Load dataset features
    dataset_folder = "images"
    dataset_features = load_dataset_features(dataset_folder)

    # Load CLIP features
    clip_dataset_features = load_clip_dataset_features(dataset_folder, clip_model)

    # Load Base CNN features
    base_cnn_dataset_features = load_base_cnn_dataset_features(dataset_folder, base_cnn_model)

    # User selection for viewing results
    option = st.selectbox("Select matching method:", ["SimCLR + DeiT", "CLIP", "Base CNN", "All Models"])

    if option == "SimCLR + DeiT":
        most_similar_images = find_most_similar_image(uploaded_features, dataset_features, method='all')
    elif option == "CLIP":
        uploaded_clip_feature = extract_clip_features(clip_model, uploaded_image).reshape(1, -1)
        most_similar_images = find_most_similar_image(uploaded_features, dataset_features, clip_features=clip_dataset_features, method='clip')
    elif option == "Base CNN":
        uploaded_base_cnn_feature = extract_base_cnn_features(base_cnn_model, uploaded_image).reshape(1, -1)
        most_similar_images = find_most_similar_image(uploaded_features, dataset_features, base_cnn_features=base_cnn_dataset_features, method='base_cnn')
    else:  # All Models (SimCLR + DeiT + CLIP)
        uploaded_clip_feature = extract_clip_features(clip_model, uploaded_image).reshape(1, -1)
        most_similar_images = find_most_similar_image(uploaded_features, dataset_features, method='all')
        
    st.subheader("Most Similar Images:")
    for img_path, score in most_similar_images[:5]:  # Show top 5 similar images
        st.image(img_path, caption=f'Similarity Score: {score:.4f}', use_column_width=True)
