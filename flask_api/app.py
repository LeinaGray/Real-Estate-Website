# flask_api/app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import os
from model import load_models, extract_features, find_most_similar_image, load_dataset_features, combine_similarities

app = Flask(__name__)
CORS(app)

# Load models when the server starts, including clip_processor
simclr_model, deit_model, clip_model, clip_processor, base_cnn_model = load_models()  # Now unpacking five variables

# Folder paths for images and features
IMAGE_FOLDER = r"C:\images"
SIMCLR_FEATURES_FILE = 'simclr_features.pkl'
DEIT_FEATURES_FILE = 'deit_features.pkl'
CLIP_FEATURES_FILE = 'clip2_features.pkl'
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

# app.py

@app.route('/api/search-by-image', methods=['POST'])
def search_by_image():
    # Get the uploaded image from the request
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Get the selected model from the request
    model_choice = request.form.get('model')
    if not model_choice:
        return jsonify({"error": "No model selected"}), 400

    # Load the uploaded image
    image = Image.open(file).convert("RGB")

    # Use the selected model for feature extraction
    if model_choice == 'simclr':
        uploaded_features = extract_features(simclr_model, None, image, 'simclr')
        dataset_features = load_dataset_features(IMAGE_FOLDER, simclr_model, 'simclr', SIMCLR_FEATURES_FILE)
    elif model_choice == 'deit':
        uploaded_features = extract_features(deit_model, None, image, 'deit')
        dataset_features = load_dataset_features(IMAGE_FOLDER, deit_model, 'deit', DEIT_FEATURES_FILE)
    elif model_choice == 'clip':
        uploaded_features = extract_features(clip_model, clip_processor, image, 'clip')
        dataset_features = load_dataset_features(IMAGE_FOLDER, clip_model, 'clip', CLIP_FEATURES_FILE)
    elif model_choice == 'base_cnn':
        uploaded_features = extract_features(base_cnn_model, None, image, 'base_cnn')
        dataset_features = load_dataset_features(IMAGE_FOLDER, base_cnn_model, 'base_cnn', CNN_FEATURES_FILE)
    elif model_choice == 'combined':
        uploaded_features_deit = extract_features(deit_model, None, image, 'deit')
        dataset_features_deit = load_dataset_features(IMAGE_FOLDER, deit_model, 'deit', DEIT_FEATURES_FILE)

        uploaded_features_clip = extract_features(clip_model, clip_processor, image, 'clip')
        dataset_features_clip = load_dataset_features(IMAGE_FOLDER, clip_model, 'clip', CLIP_FEATURES_FILE)

        most_similar_images_deit = find_most_similar_image(uploaded_features_deit, dataset_features_deit)
        most_similar_images_clip = find_most_similar_image(uploaded_features_clip, dataset_features_clip)

        combined_results = combine_similarities(most_similar_images_simclr, most_similar_images_deit, most_similar_images_clip)
    else:
        return jsonify({"error": "Invalid model choice"}), 400

    # For non-combined models, find similar images using the selected model's features
    if model_choice != 'combined':
        most_similar_images = find_most_similar_image(uploaded_features, dataset_features)
        result_data = [
            {
                "url": os.path.basename(img_path), 
                "similarity": f"{similarity * 100:.2f}",
                "listing_id": index  # <-- Include the listing index here
            }
            for index, (img_path, similarity) in enumerate(most_similar_images[:5])  # Track the index
        ]
    else:
        result_data = [
            {
                "url": os.path.basename(img_path), 
                "similarity": f"{similarity * 100:.2f}",
                "listing_id": index  # <-- Include the listing index here
            }
            for index, (img_path, similarity) in enumerate(combined_results[:5])  # Track the index
        ]

    return jsonify({"similarImages": result_data})

if __name__ == "__main__":
    app.run(debug=True)  
