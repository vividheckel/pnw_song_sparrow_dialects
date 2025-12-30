import numpy as np
import tensorflow as tf
from load_audio import generate_data
from tensorflow.keras.metrics import MeanIoU

model = tf.keras.models.load_model(
"predictor.keras",
compile=False
)

spectrogram_inputs, masks = generate_data()
spectrogram_inputs = np.expand_dims(spectrogram_inputs, axis=-1)

masks = np.array(masks)

preds = model.predict(spectrogram_inputs, batch_size=32)

preds = preds.reshape((-1, 308 * 775))
masks = masks.reshape((-1, 308 * 775))

preds_bin = (preds > 0.5).astype(np.int32)

miou = MeanIoU(num_classes=2)
miou.update_state(masks, preds_bin)
mean_iou_value = miou.result().numpy()

print(f"Mean IoU: {mean_iou_value:.4f}")
