import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import BinaryCrossentropy

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label
import pathlib

bce = BinaryCrossentropy()
@register_keras_serializable()
def weighted_loss(y_true, y_pred):
    weights = tf.where(tf.equal(y_true, 1), 5.0, 1.0)
    return tf.reduce_mean(weights * bce(y_true, y_pred))

loaded_model = load_model('predictor.keras', custom_objects={'weighted_loss': weighted_loss})

def tapered_frequency_mask(spectrogram, high_cut=6000, sr=22050, taper_width=60):
    freq_bins, _ = spectrogram.shape
    frequencies = np.linspace(0, sr / 2, freq_bins)

    mask = np.zeros_like(frequencies)
    high_idx = np.argmax(frequencies >= high_cut)
    mask[:high_idx] = 1
    
    if high_idx + taper_width < len(frequencies):
        mask[high_idx:high_idx + taper_width] = np.linspace(1, 0, taper_width)
    
    spectrogram_tapered = spectrogram * mask[:, np.newaxis]
    return spectrogram_tapered

base_path = "processed_spectrograms/"

def fill_gaps(binary_mask):
    output_mask = np.zeros_like(binary_mask)

    for i, row in enumerate(binary_mask):
        labeled_row, num_features = label(row)
        start_indices = []
        end_indices = []
        for feature_num in range(1, num_features + 1):
            component = np.where(labeled_row == feature_num)[0]
            if len(component) > 0:
                start_indices.append(component[0])
                end_indices.append(component[-1])        
        for j in range(len(start_indices) - 1):
            gap = start_indices[j+1] - end_indices[j]
            if gap <= 20:
                output_mask[i, end_indices[j]:start_indices[j+1]] = 1
                
    for i, row in enumerate(binary_mask):
        labeled_row, num_features = label(row)
        for feature_num in range(1, num_features + 1):
            component = np.where(labeled_row == feature_num)[0]
            if len(component) > 90:
                output_mask[i, component] = 1

    for j in range(output_mask.shape[1]):
        if np.any(binary_mask[:, j] == 1):
            output_mask[:, j] = 1

    return output_mask

def generate_mask_predictions(image_path):
    spectrogram_image = load_img(image_path, color_mode="grayscale", target_size=(308, 775))
    spectrogram_image = img_to_array(spectrogram_image)
    spectrogram_image = spectrogram_image / 255.0

    spectrogram_image = cv2.GaussianBlur(spectrogram_image, (5, 5), 1)
    spectrogram_image = tapered_frequency_mask(spectrogram_image)

    input_data = spectrogram_image.reshape(1, 308, 775, 1)

    predictions = loaded_model.predict(input_data)
    pred_values = predictions.reshape(308, 775)
    threshold = 0.5
    binary_mask = (pred_values > threshold).astype(np.uint8)

    output_mask = fill_gaps(binary_mask)
    output_mask = fill_gaps(output_mask)

    return output_mask
    
image_paths = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.lower().endswith('.png'):
            image_paths.append(os.path.join(root, file))

import random
import pandas as pd

LAT_MIN, LAT_MAX = 45.5, 49.5
LON_MIN, LON_MAX = -124.5, -116.0

df = pd.read_csv('macaulay.csv')
df['Latitude'] = pd.to_numeric(df['lat'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['lng'], errors='coerce')
df['Asset ID'] = pd.to_numeric(df['id'], errors='coerce')
df.dropna(subset=['Asset ID','Latitude','Longitude'], inplace=True)

id_to_coords = { str(int(r['Asset ID'])): (r['Latitude'], r['Longitude']) for _, r in df.iterrows() }

image_paths = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.lower().endswith('.png'):
            full_path = os.path.join(root, file)
            parts = full_path.split(os.sep)
            asset_id = parts[-2] if len(parts) >= 2 else None

            if asset_id in id_to_coords:
                lat, lon = id_to_coords[asset_id]
                if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                    image_paths.append(full_path)

for image_path in image_paths:
    output_mask = generate_mask_predictions(image_path)

    spectrogram_image = load_img(image_path, color_mode="grayscale", target_size=(308, 775))
    spectrogram_image = img_to_array(spectrogram_image) / 255.0
    spectrogram_image = spectrogram_image.reshape(308, 775)

    masked_spectrogram = spectrogram_image * output_mask
    masked_spectrogram_uint8 = (masked_spectrogram * 255).astype(np.uint8)

    full_stem = pathlib.Path(image_path).stem
    asset_id = full_stem.split('_')[0]
    
    save_dir = f"masked/{asset_id}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{full_stem}.png")
    black_pixel_ratio = np.mean(masked_spectrogram_uint8 == 0)

    if black_pixel_ratio > 0.95:
        continue 

    cv2.imwrite(save_path, masked_spectrogram_uint8)
