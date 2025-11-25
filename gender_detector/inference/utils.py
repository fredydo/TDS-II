import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog

IMG_SIZE = 64
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def extract_hog_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    features = hog(img_resized,
                   orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys',
                   feature_vector=True)
    return features

def predict(model, scaler, image_path):
    features = extract_hog_features(image_path)
    if features is None:
        print(f"Failed to extract features from {image_path}")
        return None

    features = scaler.transform([features])
    pred = model.predict(features)[0]
    prob  = model.predict_proba(features)[0]
    label = "Male" if pred == 0 else "Female"

    print(f"Probabilities → Male: {prob[0]:.2f}, Female: {prob[1]:.2f}")
    return label, prob