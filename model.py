import pandas as pd
import numpy as np
import tensorflow as tf
import json
from tensorflow.python.client import device_lib

NUM_LOCATIONS = 5  # Number of locations to select
MIN_CALLS_PER_LOCATION = 50  # Minimum calls required to create a location
RADIUS_KM = 8.0 # Region radius
MIN_SEPARATION_KM = 150  # Minimum distance between locations
MAX_SEPARATION_KM = 800  # Maximum distance between locations

REGION_BOUNDS = {
    'lat_min': 5,
    'lat_max': 75,
    'lon_min': -170,
    'lon_max': -30
}

df = pd.read_csv('data.csv') # Load metadata about files 

df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
df['Asset ID'] = pd.to_numeric(df['Asset ID'], errors='coerce')
df = df.dropna(subset=['Asset ID', 'Latitude', 'Longitude', 'Month'])

df = df[
    (df['Latitude'] >= REGION_BOUNDS["lat_min"]) &
    (df['Latitude'] <= REGION_BOUNDS["lat_max"]) &
    (df['Longitude'] >= REGION_BOUNDS["lon_min"]) &
    (df['Longitude'] <= REGION_BOUNDS["lon_max"])
]

# Calculate distance between two points using haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

# Find locations with sufficient density
def find_dense_locations(df, num_locations, min_calls, radius_km, min_separation_km=150, max_separation_km=800):
    unique_coords = df[['Latitude', 'Longitude']].drop_duplicates().values    
    location_densities = []
    for i, (lat, lon) in enumerate(unique_coords):
        distances = haversine_distance(
            lat, lon,
            df['Latitude'].values,
            df['Longitude'].values
        )
        calls_in_radius = np.sum(distances <= radius_km)
        if calls_in_radius >= min_calls:
            location_densities.append({
                'lat': lat,
                'lon': lon,
                'num_calls': calls_in_radius,
                'asset_ids': df[distances <= radius_km]['Asset ID'].tolist()
            })
    if len(location_densities) == 0:
        print(f"\nNo locations found with >={min_calls} calls")
        return []
    
    location_densities.sort(key=lambda x: x['num_calls'], reverse=True)
        
    selected_locations = []
    
    top_candidates = location_densities[:max(1, len(location_densities) // 5)]
    first_location = np.random.choice(top_candidates)
    selected_locations.append(first_location)
    
    for i in range(1, num_locations):
        valid_candidates = []
        
        for candidate in location_densities:
            if candidate in selected_locations:
                continue
            
            distances_to_selected = []
            too_close = False
            too_far = True
            
            for selected in selected_locations:
                dist = haversine_distance(
                    candidate['lat'], candidate['lon'],
                    selected['lat'], selected['lon']
                )
                distances_to_selected.append(dist)
                
                if dist < min_separation_km:
                    too_close = True
                    break
                
                if dist <= max_separation_km:
                    too_far = False
            
            if not too_close and not too_far:
                min_dist = min(distances_to_selected)
                score = candidate['num_calls'] * (1.0 + 0.1 * min(min_dist / 100.0, 5.0))
                valid_candidates.append((candidate, score))
        
        if len(valid_candidates) == 0:
            for candidate in location_densities:
                if candidate in selected_locations:
                    continue
                
                min_dist_to_selected = float('inf')
                too_close = False
                
                for selected in selected_locations:
                    dist = haversine_distance(
                        candidate['lat'], candidate['lon'],
                        selected['lat'], selected['lon']
                    )
                    min_dist_to_selected = min(min_dist_to_selected, dist)
                    
                    if dist < min_separation_km:
                        too_close = True
                        break
                
                if not too_close and min_dist_to_selected < float('inf'):
                    score = candidate['num_calls']
                    valid_candidates.append((candidate, score))
        
        if len(valid_candidates) == 0:
            break
        
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        top_valid = valid_candidates[:max(1, len(valid_candidates) // 3)]
        
        candidates_only = [c for c, s in top_valid]
        scores = np.array([s for c, s in top_valid])
        probabilities = scores / scores.sum()
        
        chosen = np.random.choice(candidates_only, p=probabilities)
        
        separations = [
            haversine_distance(chosen['lat'], chosen['lon'], sel['lat'], sel['lon'])
            for sel in selected_locations
        ]
        min_sep = min(separations)
        
        selected_locations.append(chosen)
    
    if len(selected_locations) > 1:
        print(f"\n=== Geographic Distribution Summary ===")
        lat_range = max(loc['lat'] for loc in selected_locations) - min(loc['lat'] for loc in selected_locations)
        lon_range = max(loc['lon'] for loc in selected_locations) - min(loc['lon'] for loc in selected_locations)
        print(f"Latitude range: {lat_range:.2f}°")
        print(f"Longitude range: {lon_range:.2f}°")
        
        distances = []
        for i, loc1 in enumerate(selected_locations):
            for loc2 in selected_locations[i+1:]:
                dist = haversine_distance(loc1['lat'], loc1['lon'], loc2['lat'], loc2['lon'])
                distances.append(dist)
    return selected_locations

selected_locations = find_dense_locations(df, NUM_LOCATIONS, MIN_CALLS_PER_LOCATION, RADIUS_KM, MIN_SEPARATION_KM, MAX_SEPARATION_KM)

if len(selected_locations) == 0:
    raise ValueError(f"No locations found with at least {MIN_CALLS_PER_LOCATION} calls within {RADIUS_KM}km radius")

selected_asset_ids = set()
for loc in selected_locations:
    selected_asset_ids.update(loc['asset_ids'])

df_filtered = df[df['Asset ID'].isin(selected_asset_ids)].copy()

id_to_coords = {
    str(row['Asset ID']): (row['Latitude'], row['Longitude'])
    for _, row in df_filtered.iterrows()
}

lat_mean = df_filtered["Latitude"].mean()
lat_std = df_filtered["Latitude"].std()
lon_mean = df_filtered["Longitude"].mean()
lon_std = df_filtered["Longitude"].std()

import os
from glob import glob

image_paths = []
labels = []
base_dir = 'data/unmasked/' # Data with spectrogram images

for asset_id in os.listdir(base_dir):
    asset_id_clean = asset_id.strip()

    if asset_id_clean not in id_to_coords:
        continue

    folder = os.path.join(base_dir, asset_id_clean)
    if not os.path.isdir(folder):
        continue

    latlon = id_to_coords[asset_id_clean]
    
    mon = df_filtered.loc[df_filtered['Asset ID'] == float(asset_id_clean), 'Month'].values
    if len(mon) == 0 or pd.isna(mon[0]):
        continue
    mon = mon[0]

    img_files = sorted(glob(os.path.join(folder, '*.png')))[:-1]
    for img_file in img_files:
        lat, lon = latlon
        if pd.isna(lat) or pd.isna(lon) or pd.isna(mon):
            continue
        image_paths.append(img_file)
        month_rad = 2 * np.pi * (float(mon) - 1.0) / 12.0
        month_sin = np.sin(month_rad)
        month_cos = np.cos(month_rad)

        lat_norm = (float(lat) - lat_mean) / lat_std
        lon_norm = (float(lon) - lon_mean) / lon_std
        labels.append((lat_norm, lon_norm, month_sin, month_cos))

# Create gaussian kernel for filtering
def create_gaussian_kernel(size, sigma):
    ax = tf.range(-size // 2 + 1., size // 2 + 1.)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    return tf.reshape(kernel, [size, size, 1, 1])

# Smooth frequency mask
def smooth_frequency_mask(image, freq_range, transition_width=0.05):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    
    freq_coords = tf.linspace(0., 1., tf.cast(h, tf.int32))
    freq_coords = tf.reshape(freq_coords, [-1, 1])
    
    low_freq, high_freq = freq_range
    
    lower_mask = tf.sigmoid((freq_coords - low_freq) / transition_width)
    upper_mask = tf.sigmoid((high_freq - freq_coords) / transition_width)
    
    freq_mask = lower_mask * upper_mask
    freq_mask = tf.broadcast_to(freq_mask, tf.shape(image)[:2])
    freq_mask = tf.expand_dims(freq_mask, -1)
    
    return freq_mask

# Enhance regions with higher energy
def adaptive_energy_enhancement(spectrogram, percentile_threshold=70):
    energy = tf.reduce_mean(tf.square(spectrogram), axis=[1, 2], keepdims=True)
    
    energy_flat = tf.reshape(energy, [-1])
    energy_sorted = tf.sort(energy_flat)
    n_elements = tf.shape(energy_sorted)[0]
    threshold_idx = tf.cast(tf.cast(n_elements, tf.float32) * percentile_threshold / 100.0, tf.int32)
    threshold = energy_sorted[threshold_idx]
    
    enhancement = tf.sigmoid((energy - threshold) * 10.0)
    enhancement = tf.clip_by_value(enhancement, 0.5, 2.0)
    
    enhancement = tf.broadcast_to(enhancement, tf.shape(spectrogram))
    
    return spectrogram * enhancement

# Enhance harmonic content
def harmonic_enhancement(spectrogram, fundamental_range=(0.1, 0.6)):
    h = tf.cast(tf.shape(spectrogram)[0], tf.float32)
    
    freq_coords = tf.linspace(0., 1., tf.cast(h, tf.int32))
    
    fund_low, fund_high = fundamental_range
    harmonic_weight = tf.exp(-tf.square((freq_coords - (fund_low + fund_high) / 2) / 0.2))
    harmonic_weight = 1.0 + 0.5 * harmonic_weight
    
    harmonic_weight = tf.reshape(harmonic_weight, [-1, 1, 1])
    return spectrogram * harmonic_weight

# Smooth noise gating (background noise supression)
def noise_gate(spectrogram, gate_threshold=0.1, smooth_width=0.02):
    energy = tf.reduce_mean(tf.square(spectrogram), axis=-1, keepdims=True)
    
    energy_min = tf.reduce_min(energy)
    energy_max = tf.reduce_max(energy)
    energy_norm = (energy - energy_min) / (energy_max - energy_min + 1e-8)
    
    gate = tf.sigmoid((energy_norm - gate_threshold) / smooth_width)
    
    return spectrogram * gate

# Load image and apply preprocessing and augmentation
def load_image(regular_path, label, apply_augment=True):
    masked_path = tf.strings.regex_replace(regular_path, "/unmasked/", "masked/")
    
    reg_img = tf.io.read_file(regular_path)
    reg_img = tf.image.decode_png(reg_img, channels=3)
    reg_img = tf.image.resize(reg_img, [224, 224])
    reg_img = tf.cast(reg_img, tf.float32) / 255.0

    masked_img = tf.io.read_file(masked_path)
    masked_img = tf.image.decode_png(masked_img, channels=1)
    masked_img = tf.image.resize(masked_img, [224, 224])
    masked_img = tf.cast(masked_img, tf.float32) / 255.0
    
    gaussian_kernel = create_gaussian_kernel(5, 1.0)
    masked_img_4d = tf.expand_dims(masked_img, 0)
    masked_img = tf.nn.conv2d(masked_img_4d, gaussian_kernel, 
                             strides=[1, 1, 1, 1], padding='SAME')[0]
    masked_img_rgb = tf.image.grayscale_to_rgb(masked_img)

    if apply_augment:
        freq_low = tf.random.uniform([], 0.15, 0.25)
        freq_high = tf.random.uniform([], 0.75, 0.90)
    else:
        freq_low = 0.2
        freq_high = 0.8
    
    freq_mask = smooth_frequency_mask(reg_img, (freq_low, freq_high), transition_width=0.08)
    reg_img = reg_img * freq_mask
    
    reg_img = adaptive_energy_enhancement(reg_img)
    reg_img = harmonic_enhancement(reg_img)
    
    channels = tf.unstack(reg_img, axis=-1)
    smoothed_channels = []
    
    for channel in channels:
        channel = tf.expand_dims(channel, -1)
        kernel_1d = tf.ones([1, 3, 1, 1], dtype=tf.float32) / 3.0
        channel_4d = tf.expand_dims(channel, 0)
        smoothed = tf.nn.conv2d(channel_4d, kernel_1d, strides=[1, 1, 1, 1], padding='SAME')
        smoothed_channels.append(tf.squeeze(smoothed, [0, -1]))
    
    reg_img = tf.stack(smoothed_channels, axis=-1)
    
    reg_img = noise_gate(reg_img, gate_threshold=0.12)
    
    reg_img = tf.image.resize(reg_img, [224, 224])
    masked_img_rgb = tf.image.resize(masked_img_rgb, [224, 224])
    
    if apply_augment:
        noise_level = tf.random.uniform([], 0.005, 0.02)
    else:
        noise_level = 0.01
    
    noise = tf.random.normal(tf.shape(reg_img), 0.0, noise_level)
    channels = tf.unstack(noise, axis=-1)
    smoothed_noise_channels = []
    
    smooth_noise_kernel = create_gaussian_kernel(3, 0.5)
    for channel in channels:
        channel = tf.expand_dims(channel, -1)
        channel_4d = tf.expand_dims(channel, 0)
        smoothed = tf.nn.conv2d(channel_4d, smooth_noise_kernel, 
                               strides=[1, 1, 1, 1], padding='SAME')
        smoothed_noise_channels.append(tf.squeeze(smoothed, [0, -1]))
    
    noise = tf.stack(smoothed_noise_channels, axis=-1)
    
    alpha = masked_img_rgb
    alpha_channels = tf.unstack(alpha, axis=-1)
    smoothed_alpha_channels = []
    
    alpha_smooth_kernel = create_gaussian_kernel(5, 1.5)
    for channel in alpha_channels:
        channel = tf.expand_dims(channel, -1)
        channel_4d = tf.expand_dims(channel, 0)
        smoothed = tf.nn.conv2d(channel_4d, alpha_smooth_kernel, 
                               strides=[1, 1, 1, 1], padding='SAME')
        smoothed_alpha_channels.append(tf.squeeze(smoothed, [0, -1]))
    
    alpha = tf.stack(smoothed_alpha_channels, axis=-1)
    
    reg_img = reg_img * alpha + noise * (1.0 - alpha)
    
    if apply_augment:
        if tf.random.uniform([]) > 0.8:
            mask_freq_low = tf.random.uniform([], 0.0, 0.2)
            mask_freq_high = tf.random.uniform([], 0.8, 1.0)
            
            mask_low = smooth_frequency_mask(reg_img, (0.0, mask_freq_low), 0.05)
            mask_high = smooth_frequency_mask(reg_img, (mask_freq_high, 1.0), 0.05)
            
            reg_img = reg_img * (1.0 - mask_low * 0.8)
            reg_img = reg_img * (1.0 - mask_high * 0.6)
        
        if tf.random.uniform([]) > 0.85:
            stretch_factor = tf.random.uniform([], 0.9, 1.1)
            new_width = tf.cast(tf.cast(tf.shape(reg_img)[1], tf.float32) * stretch_factor, tf.int32)
            reg_img = tf.image.resize(reg_img, [224, new_width], 
                                    method=tf.image.ResizeMethod.BILINEAR)
            reg_img = tf.image.resize_with_crop_or_pad(reg_img, 224, 224)
        
        if tf.random.uniform([]) > 0.85:
            shift_amount = tf.random.uniform([], -0.1, 0.1)
            shift_pixels = tf.cast(shift_amount * tf.cast(tf.shape(reg_img)[0], tf.float32), tf.int32)
            
            reg_img = tf.roll(reg_img, shift_pixels, axis=0)
            
            if tf.abs(shift_pixels) > 0:
                boundary_size = tf.minimum(tf.abs(shift_pixels), 10)
                fade = tf.linspace(0., 1., boundary_size)
                fade = tf.reshape(fade, [-1, 1, 1])
                
                if shift_pixels > 0:
                    reg_img = tf.concat([
                        reg_img[:boundary_size] * fade,
                        reg_img[boundary_size:]
                    ], axis=0)
                else:
                    reg_img = tf.concat([
                        reg_img[:-boundary_size],
                        reg_img[-boundary_size:] * tf.reverse(fade, [0])
                    ], axis=0)

    channels = tf.unstack(reg_img, axis=-1)
    final_channels = []
    
    final_smooth_kernel = create_gaussian_kernel(3, 0.3)
    for channel in channels:
        channel = tf.expand_dims(channel, -1)
        channel_4d = tf.expand_dims(channel, 0)
        smoothed = tf.nn.conv2d(channel_4d, final_smooth_kernel, 
                              strides=[1, 1, 1, 1], padding='SAME')
        final_channels.append(tf.squeeze(smoothed, [0, -1]))
    
    reg_img = tf.stack(final_channels, axis=-1)
    
    reg_img = tf.image.resize(reg_img, [224, 224])
    reg_img = tf.clip_by_value(reg_img, 0.0, 1.0)
    
    label = tf.cast(label, tf.float32)
    lat, lon, mon_sin, mon_cos = tf.unstack(label)
    label_coords = tf.stack([lat, lon])
    month_input = tf.stack([mon_sin, mon_cos])

    return {"image_input": reg_img, "month_input": month_input}, label_coords

filtered_image_paths = []
filtered_labels = []
filtered_locations = []

asset_to_location = {}
for loc_idx, loc in enumerate(selected_locations):
    for asset_id in loc['asset_ids']:
        asset_to_location[asset_id] = loc_idx

for path, label in zip(image_paths, labels):
    masked = path.replace("unmasked/", "masked/")
    if os.path.exists(masked):
        asset_id_str = path.split('/')[-2]
        try:
            asset_id = float(asset_id_str)
            if asset_id in asset_to_location:
                filtered_image_paths.append(path)
                filtered_labels.append(label)
                filtered_locations.append(asset_to_location[asset_id])
        except ValueError:
            continue

location_counts = {}
for loc_idx in filtered_locations:
    location_counts[loc_idx] = location_counts.get(loc_idx, 0) + 1

image_paths_tf = tf.constant(filtered_image_paths, dtype=tf.string)
labels_tf = tf.constant(filtered_labels, dtype=tf.float32)

paths_ds = tf.data.Dataset.from_tensor_slices((image_paths_tf, labels_tf))
ds = paths_ds.map(lambda p, l: load_image(p, l), num_parallel_calls=tf.data.AUTOTUNE)

# Start machine learning process
from sklearn.model_selection import train_test_split

train_paths, test_paths, train_labels, test_labels, train_locs, test_locs = train_test_split(
    filtered_image_paths, filtered_labels, filtered_locations, 
    test_size=0.2, random_state=42, stratify=filtered_locations
)
val_paths, test_paths, val_labels, test_labels, val_locs, test_locs = train_test_split(
    test_paths, test_labels, test_locs,
    test_size=0.5, random_state=42, stratify=test_locs
)

# Jitter labels (augmentation)
def jitter_labels(inputs, labels):
    reg_coords = labels['regression_output']
    lat, lon = tf.split(reg_coords, 2)
    lat += tf.random.uniform([], -0.01, 0.01)
    lon += tf.random.uniform([], -0.01, 0.01)
    jittered_coords = tf.concat([lat, lon], axis=-1)
    return inputs, {'regression_output': jittered_coords, 'class_output': labels['class_output']}

# Spectrogram augmentation
def augment_image(inputs):
    img = inputs["image_input"]
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    
    inputs["image_input"] = img
    return inputs

# Prepare dataset for ML
def prepare_dataset(paths, regression_labels, location_labels, batch_size=32, shuffle=True, augment=False):
    paths_tf = tf.constant(paths, dtype=tf.string)
    regression_labels_tf = tf.constant(regression_labels, dtype=tf.float32)
    location_labels_tf = tf.constant(location_labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths_tf, regression_labels_tf, location_labels_tf))

    def map_fn(path, reg_label, loc_label):
        inputs, coords = load_image(path, reg_label, apply_augment=augment)
        return inputs, {'regression_output': coords, 'class_output': loc_label}

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(jitter_labels, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

# Calculate mean haversine error
def mean_haversine_km(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    tf.debugging.assert_shapes([
        (y_true, ('batch', 2)),
        (y_pred, ('batch', 2))
    ], message="y_true or y_pred doesn't have shape (?, 2)")

    lat1, lon1 = tf.split(y_true, 2, axis=-1)
    lat2, lon2 = tf.split(y_pred, 2, axis=-1)

    lat1 = lat1 * lat_std + lat_mean
    lon1 = lon1 * lon_std + lon_mean
    lat2 = lat2 * lat_std + lat_mean
    lon2 = lon2 * lon_std + lon_mean

    lat1 = lat1 * (np.pi / 180.0)
    lon1 = lon1 * (np.pi / 180.0)
    lat2 = lat2 * (np.pi / 180.0)
    lon2 = lon2 * (np.pi / 180.0)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = tf.sin(dlat / 2.0)**2 + tf.cos(lat1) * tf.cos(lat2) * tf.sin(dlon / 2.0)**2
    c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1.0 - a))
    r = 6371.0

    dist = r * c
    return tf.reduce_mean(dist)

train_ds = prepare_dataset(train_paths, train_labels, train_locs, augment=True)
val_ds   = prepare_dataset(val_paths, val_labels, val_locs, shuffle=False)
test_ds  = prepare_dataset(test_paths, test_labels, test_locs, shuffle=False)

from tensorflow.keras import models, layers, regularizers, Input
from tensorflow.keras.layers import Lambda

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu='local')
    strategy = tf.distribute.TPUStrategy(tpu)
except (ValueError, tf.errors.NotFoundError):
    if len(tf.config.list_physical_devices('GPU')) > 0:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

with strategy.scope():
    
    img_input = Input(shape=(224, 224, 3), name="image_input")
    month_input = Input(shape=(2,), name="month_input")
    
    x = layers.Conv2D(32, 7, strides=2, padding='same')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    def residual_block(x, filters, strides=1):
        shortcut = x
        
        x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=strides)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x
    
    x = residual_block(x, 32, strides=1)
    x = residual_block(x, 32)
    x = residual_block(x, 64, strides=2)
    x = residual_block(x, 64)
    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 128)
    
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(128 // 16, activation='relu')(attention)
    attention = layers.Dense(128, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, 128))(attention)
    x = layers.Multiply()([x, attention])
    
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    
    adaptive_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2]))(x)
    
    combined_features = layers.Concatenate()([gap, gmp, adaptive_pool])
    
    month_processed = layers.Dense(32, activation='relu')(month_input)
    month_processed = layers.Dense(64, activation='relu')(month_processed)
    
    combined = layers.Concatenate()([combined_features, month_processed])
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(combined)
    x = layers.Dropout(0.5)(x)
    x_res = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.4)(x_res)
    
    x_skip = layers.Dense(128)(combined)
    x = layers.Add()([x, x_skip])
    x = layers.ReLU()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    reg_output = layers.Dense(2, name="regression_output")(x)
    class_output = layers.Dense(NUM_LOCATIONS, activation='softmax', name="class_output")(x)
    
    model = tf.keras.Model(inputs=[img_input, month_input], outputs=[reg_output, class_output])

    import tensorflow.keras.backend as K    
    current_epoch_var = tf.Variable(0, trainable=False, dtype=tf.int32)

    optimizer = tf.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'regression_output': mean_haversine_km,
            'class_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'regression_output': 0.6,
            'class_output': 0.4,
        },
        metrics={
            'regression_output': [mean_haversine_km],
            'class_output': ['accuracy']
        }
    )

    class EpochTracker(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            current_epoch_var.assign(epoch)

    # Learning rate scheduler and early stopping
    def scheduler(epoch, lr):
        warmup_epochs = 10
        base_lr = 1e-4
    
        if epoch < warmup_epochs:
            return 1e-6 + (base_lr - 1e-6) * (epoch / warmup_epochs)
        elif epoch < 150:
            return base_lr
        elif epoch < 250:
            return base_lr * 0.5
        else:
            return base_lr * 0.1
    
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    EPOCHS_TOTAL = 150
    CHUNK_SIZE = 10
    
    for start in range(0, EPOCHS_TOTAL, CHUNK_SIZE):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=start,
            epochs=start + CHUNK_SIZE,
            verbose=1,
            callbacks=[
                EpochTracker(),
                lr_scheduler,
                early_stop
            ]
        )

        with open(f'working/history_epoch_{start+CHUNK_SIZE}.json', 'w') as f:
            json.dump(history.history, f)
    
        model.save(f"working/model_epoch_{start+CHUNK_SIZE}.keras")
        
        if early_stop.stopped_epoch > 0:
            print(f"\nEarly stopping triggered at epoch {early_stop.stopped_epoch + 1}")
            break
    
    # Save model and stats
    model.save('working/model.keras')
    model.save_weights("working/model_weights.weights.h5")
    np.savez("stats.npz",
             lat_mean=lat_mean,
             lat_std=lat_std,
             lon_mean=lon_mean,
             lon_std=lon_std)
    
    location_info = {
        'selected_locations': [
            {
                'lat': float(loc['lat']),
                'lon': float(loc['lon']),
                'num_calls': int(loc['num_calls'])
            }
            for loc in selected_locations
        ],
        'config': {
            'num_locations': int(NUM_LOCATIONS),
            'min_calls_per_location': int(MIN_CALLS_PER_LOCATION),
            'radius_km': float(RADIUS_KM)
        },
        'total_samples': int(len(filtered_image_paths)),
        'total_assets': int(len(selected_asset_ids))
    }
    
    with open('working/location_info.json', 'w') as f:
        json.dump(location_info, f, indent=2)

# Testing model performance
test_predictions_reg = []
test_predictions_class = []
test_actuals_reg = []
test_actuals_class = []
test_errors_km = []

for inputs, labels in test_ds.take(10):
    preds = model.predict(inputs, verbose=0)
    test_predictions_reg.append(preds[0])
    test_predictions_class.append(preds[1])
    test_actuals_reg.append(labels['regression_output'].numpy())
    test_actuals_class.append(labels['class_output'].numpy())

test_predictions_reg = np.vstack(test_predictions_reg)
test_predictions_class = np.vstack(test_predictions_class)
test_actuals_reg = np.vstack(test_actuals_reg)
test_actuals_class = np.concatenate(test_actuals_class)

pred_classes = np.argmax(test_predictions_class, axis=1)

pred_lats = test_predictions_reg[:, 0] * lat_std + lat_mean
pred_lons = test_predictions_reg[:, 1] * lon_std + lon_mean
actual_lats = test_actuals_reg[:, 0] * lat_std + lat_mean
actual_lons = test_actuals_reg[:, 1] * lon_std + lon_mean

for i in range(len(pred_lats)):
    error_km = haversine_distance(
        actual_lats[i], actual_lons[i],
        pred_lats[i], pred_lons[i]
    )
    test_errors_km.append(error_km)
