import json
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="conv1"),
    tf.keras.layers.MaxPooling2D(pool_size=2, name="pool1"),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu", name="conv2"),
    tf.keras.layers.MaxPooling2D(pool_size=2, name="pool2"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu", name="dense1"),
    tf.keras.layers.Dense(10, activation="softmax", name="dense2"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[..., None].astype("float32") / 255.0
x_test = x_test[..., None].astype("float32") / 255.0

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
    verbose=2,
)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

weights = {
    "conv1Kernel": model.get_layer("conv1").get_weights()[0].reshape(-1).tolist(),
    "conv1Bias": model.get_layer("conv1").get_weights()[1].tolist(),
    "conv2Kernel": model.get_layer("conv2").get_weights()[0].reshape(-1).tolist(),
    "conv2Bias": model.get_layer("conv2").get_weights()[1].tolist(),
    "dense1Kernel": model.get_layer("dense1").get_weights()[0].tolist(),
    "dense1Bias": model.get_layer("dense1").get_weights()[1].tolist(),
    "dense2Kernel": model.get_layer("dense2").get_weights()[0].tolist(),
    "dense2Bias": model.get_layer("dense2").get_weights()[1].tolist(),
    "testAccuracy": float(test_accuracy),
}
(MODEL_DIR / "cnn_weights.json").write_text(
    json.dumps(weights),
    encoding="utf-8",
)

model.save(MODEL_DIR / "mnist_cnn.keras")
(MODEL_DIR / "cnn_metrics.json").write_text(
    json.dumps({
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": 5,
        "batch_size": 128,
        "architecture": "28x28x1 -> Conv32 -> Pool -> Conv64 -> Pool -> Dense64 -> Dense10",
    }, indent=2),
    encoding="utf-8",
)

print(json.dumps({"test_accuracy": test_accuracy, "weights": str(MODEL_DIR / "cnn_weights.json")}))
