#flask_api\app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import os
import torch
from model import load_models, extract_features, find_most_similar_image, load_dataset_features

app = Flask(__name__)
CORS(app)

# Load models when the server starts
simclr_model, deit_model, clip_model, base_cnn_model = load_models()

# Folder paths for images and features
IMAGE_FOLDER = r"C:\images"
SIMCLR_FEATURES_FILE = 'simclr_features.pkl'
DEIT_FEATURES_FILE = 'deit_features.pkl'
CLIP_FEATURES_FILE = 'clip_features.pkl'
CNN_FEATURES_FILE = 'cnn_features.pkl'

# Ensure image folder exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
    
@app.route('/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

@app.route('/')
def home():
    return "Welcome to the Image Search API"

@app.route('/api/search-by-image', methods=['POST'])
def search_by_image():
    # Get the uploaded image from the request
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded file
    image = Image.open(file).convert("RGB")

    # Extract features using SimCLR model
    uploaded_features_simclr = extract_features(simclr_model, image, 'simclr')
    uploaded_features_deit = extract_features(deit_model, image, 'deit')

    # Load precomputed dataset features
    dataset_features_simclr = load_dataset_features(IMAGE_FOLDER, simclr_model, 'simclr', SIMCLR_FEATURES_FILE)
    dataset_features_deit = load_dataset_features(IMAGE_FOLDER, deit_model, 'deit', DEIT_FEATURES_FILE)

    # Find similar images
    most_similar_images_simclr = find_most_similar_image(uploaded_features_simclr, dataset_features_simclr)
    most_similar_images_deit = find_most_similar_image(uploaded_features_deit, dataset_features_deit)

    # Combine the results
    combined_results = most_similar_images_simclr + most_similar_images_deit

    # Prepare response
    result_data = []
    for img_path, similarity in combined_results[:5]:  # Limit to top 5 similar images
        # Extract the filename only for the URL
        filename = os.path.basename(img_path)  # Get the filename
        result_data.append({
            "url": filename,  # Use just the filename
            "similarity": f"{similarity:.4f}"
        })

    return jsonify({"similarImages": result_data})

if __name__ == "__main__":
    app.run(debug=True)
