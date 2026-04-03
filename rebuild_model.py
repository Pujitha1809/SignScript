"""
Run this ONCE to rebuild your model into a clean local .keras file.
Place this script in your SignLanguage/ folder alongside app.py, then run:
    python rebuild_model.py
"""

import json
import tensorflow as tf
from tensorflow import keras

# ── Load the cleaned config ──────────────────────────────────────────────────
with open("model/model_config.json") as f:
    config = json.load(f)

# ── Build model manually from architecture ───────────────────────────────────
model = keras.Sequential([
    keras.Input(shape=(64, 64, 3), name="input_layer"),

    keras.layers.Conv2D(32, (3,3), padding="same", activation="relu",  name="conv2d"),
    keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,       name="batch_normalization"),
    keras.layers.Conv2D(32, (3,3), padding="same", activation="relu",  name="conv2d_1"),
    keras.layers.MaxPooling2D((2,2),                                    name="max_pooling2d"),
    keras.layers.Dropout(0.25,                                          name="dropout"),

    keras.layers.Conv2D(64, (3,3), padding="same", activation="relu",  name="conv2d_2"),
    keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,       name="batch_normalization_1"),
    keras.layers.Conv2D(64, (3,3), padding="same", activation="relu",  name="conv2d_3"),
    keras.layers.MaxPooling2D((2,2),                                    name="max_pooling2d_1"),
    keras.layers.Dropout(0.25,                                          name="dropout_1"),

    keras.layers.Conv2D(128, (3,3), padding="same", activation="relu", name="conv2d_4"),
    keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,       name="batch_normalization_2"),
    keras.layers.Conv2D(128, (3,3), padding="same", activation="relu", name="conv2d_5"),
    keras.layers.MaxPooling2D((2,2),                                    name="max_pooling2d_2"),
    keras.layers.Dropout(0.25,                                          name="dropout_2"),

    keras.layers.Flatten(                                               name="flatten"),
    keras.layers.Dense(512, activation="relu",                         name="dense"),
    keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001,       name="batch_normalization_3"),
    keras.layers.Dropout(0.5,                                           name="dropout_3"),
    keras.layers.Dense(256, activation="relu",                         name="dense_1"),
    keras.layers.Dropout(0.3,                                           name="dropout_4"),
    keras.layers.Dense(29,  activation="softmax",                      name="dense_2"),
])

# ── Load your trained weights ─────────────────────────────────────────────────
model.load_weights("model/weights.weights.h5")
print("✅ Weights loaded successfully!")

# ── Save as a clean local .keras file ────────────────────────────────────────
model.save("model/sign_model_local.keras")
print("✅ Saved to model/sign_model_local.keras — ready to use!")
