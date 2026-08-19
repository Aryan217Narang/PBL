"""
model.py
CNN architecture for NIDS Binary Intrusion Detection (Section 3.2 / Fig. 2):
  Conv1D -> BatchNorm -> Conv1D -> BatchNorm -> MaxPool1D ->
  Conv1D -> BatchNorm -> GlobalAveragePool1D ->
  Dense (256, ReLU) -> Dropout -> Dense (128, ReLU) -> Dense (2, Softmax)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_model(input_shape: tuple, n_classes: int = 2) -> tf.keras.Model:
    """
    Build the high-accuracy 1D-CNN NIDS model.

    Parameters
    ----------
    input_shape : (n_features, 1) — shape after RFE & preprocessing
    n_classes   : number of output classes (2 for Binary Intrusion Detection)

    Returns
    -------
    Compiled Keras model
    """
    inp = layers.Input(shape=input_shape, name="input")

    # ── Convolution block 1 ─────────────────────────────────────
    x = layers.Conv1D(
        filters=64, kernel_size=3, padding="same",
        activation="relu", name="conv1_1"
    )(inp)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv1D(
        filters=64, kernel_size=3, padding="same",
        activation="relu", name="conv1_2"
    )(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # ── Convolution block 2 ─────────────────────────────────────
    x = layers.Conv1D(
        filters=128, kernel_size=3, padding="same",
        activation="relu", name="conv2_1"
    )(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv1D(
        filters=128, kernel_size=3, padding="same",
        activation="relu", name="conv2_2"
    )(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ── Fully Connected Layers ──────────────────────────────────
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense1"
    )(x)
    x = layers.Dropout(0.2, name="dropout1")(x)

    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense2"
    )(x)
    x = layers.Dropout(0.2, name="dropout2")(x)

    # Output layer — Softmax
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="CNN_NIDS")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
