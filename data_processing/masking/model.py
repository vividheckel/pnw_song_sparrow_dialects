from load_audio import generate_data
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

spectrogram_inputs, masks = generate_data()
spectrogram_inputs = np.expand_dims(spectrogram_inputs, axis=-1)

datagen = ImageDataGenerator(
    width_shift_range=0.3
)

X_train, X_test, y_train, y_test = train_test_split(spectrogram_inputs, masks, test_size=0.1, random_state=42)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = y_train.reshape((-1, 308 * 775))
y_test = y_test.reshape((-1, 308 * 775))

train_gen = datagen.flow(X_train, y_train, batch_size=32)
val_gen = datagen.flow(X_test, y_test, batch_size=32)

bce = BinaryCrossentropy()
def weighted_loss(y_true, y_pred):
    weights = tf.where(tf.equal(y_true, 1), 5.0, 1.0)
    return tf.reduce_mean(weights * bce(y_true, y_pred))

initializer = tf.keras.initializers.HeNormal()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=100000, 
    decay_rate=0.96,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model = models.Sequential([
    layers.InputLayer(shape=(308, 775, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=initializer),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=initializer),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=initializer),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(308 * 775, activation='sigmoid'),
])

model.compile(optimizer=optimizer, loss=weighted_loss, metrics=[MeanIoU(num_classes=2)])
history = model.fit(train_gen, validation_data=val_gen, epochs=100)

model.save("predictor.keras")
