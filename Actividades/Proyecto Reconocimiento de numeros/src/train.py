"""Train an MNIST digit classifier and save the model and training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(output_dir: Path = Path("models"), epochs: int = 10, batch_size: int = 128) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "data").mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "notebooks").mkdir(parents=True, exist_ok=True)
    (output_dir.parent / "models").mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0).argmax(axis=1)
    matrix = confusion_matrix(y_test, predictions)

    model_path = output_dir / "mnist_digits.keras"
    model.save(model_path)
    np.save(output_dir / "confusion_matrix.npy", matrix)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    correct = np.flatnonzero(predictions == y_test)[:5]
    incorrect = np.flatnonzero(predictions != y_test)[:5]
    for label, indexes in (("correct", correct), ("incorrect", incorrect)):
        for position, index in enumerate(indexes):
            figure, axis = plt.subplots(figsize=(2, 2))
            axis.imshow(x_test[index].reshape(28, 28), cmap="gray")
            axis.set_title(f"real={y_test[index]} pred={predictions[index]}")
            axis.axis("off")
            figure.tight_layout()
            figure.savefig(examples_dir / f"{label}_{position + 1}.png", dpi=120)
            plt.close(figure)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "epochs": epochs,
                "batch_size": batch_size,
            },
            file,
            indent=2,
        )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="entrenamiento")
    axes[0].plot(history.history["val_loss"], label="validación")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="entrenamiento")
    axes[1].plot(history.history["val_accuracy"], label="validación")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Época")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(figure)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Ejemplos de predicción (real -> predicha):")
    for index in range(10):
        print(f"  {int(y_test[index])} -> {int(predictions[index])}")

    figure, axis = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(matrix).plot(ax=axis, cmap="Blues", colorbar=False)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    return {"model": str(model_path), "test_accuracy": float(test_accuracy)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    arguments = parser.parse_args()
    print(train(arguments.output_dir, arguments.epochs, arguments.batch_size))
