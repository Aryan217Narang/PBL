"""
model.py
CNN architecture from paper Section 3.2 / Fig. 2:
  Conv1D → MaxPooling1D → Conv1D → MaxPooling1D →
  Flatten → Dense (ReLU) → Dropout → Dense (Softmax)

Layers: 1 Input + 3 Hidden + 1 Output
Neurons per layer: 60,000 (approximated via filter counts)
Activation hidden: ReLU
Activation output: Softmax
Optimizer: Adam
Loss: sparse_categorical_crossentropy
Batch size: 32  |  Epochs: 10
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cnn_model(input_shape: tuple, n_classes: int) -> tf.keras.Model:
    """
    Build the CNN-based NIDS model as described in the paper.

    Parameters
    ----------
    input_shape : (n_features, 1)   — shape after preprocessing reshape
    n_classes   : number of output classes

    Returns
    -------
    Compiled Keras model
    """
    inp = layers.Input(shape=input_shape, name="input")

    # ── Convolution block 1 ─────────────────────────────────────
    x = layers.Conv1D(
        filters=64, kernel_size=3, padding="same",
        activation="relu", name="conv1"
    )(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # ── Convolution block 2 ─────────────────────────────────────
    x = layers.Conv1D(
        filters=128, kernel_size=3, padding="same",
        activation="relu", name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # ── Flatten + Fully Connected ────────────────────────────────
    x = layers.Flatten(name="flatten")(x)

    # Hidden layer 1  (approximates 60,000 neurons from paper)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense1"
    )(x)
    x = layers.Dropout(0.3, name="dropout1")(x)

    # Hidden layer 2
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense2"
    )(x)
    x = layers.Dropout(0.3, name="dropout2")(x)

    # Hidden layer 3
    x = layers.Dense(
        64, activation="relu", name="dense3"
    )(x)

    # Output layer — Softmax
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inp, outputs=out, name="CNN_NIDS")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model
