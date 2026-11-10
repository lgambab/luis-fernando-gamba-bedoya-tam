"""Prediction helpers for numbers from 0 to 99."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from PIL import Image

from .preprocess import ImageValidationError, prepare_input


def load_model(path: str | Path = "models/mnist_digits.keras") -> tf.keras.Model:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo {model_path}. Ejecuta primero: python -m src.train"
        )
    return tf.keras.models.load_model(model_path)


def predict_number(
    image: Image.Image | str | Path,
    model: tf.keras.Model,
    expected_digits: int | None = None,
) -> Dict[str, object]:
    """Predict one digit (0-9) or two digits (10-99).

    The returned probabilities are ordered by class 0 through 9 for each
    detected digit. General confidence is the mean confidence of the digits.
    """
    inputs = prepare_input(image, expected_digits=expected_digits)
    if not inputs or len(inputs) > 2:
        raise ImageValidationError("La imagen debe contener uno o dos dígitos.")

    probabilities: List[List[float]] = []
    for vector in inputs:
        output = np.asarray(model.predict(vector, verbose=0))
        if output.shape != (1, 10) or not np.isfinite(output).all():
            raise ImageValidationError(
                "El modelo no devolvió probabilidades válidas para las 10 clases."
            )
        probabilities.append(output[0].astype(float).tolist())

    digits = [int(np.argmax(values)) for values in probabilities]
    confidences = [float(np.max(values)) for values in probabilities]
    if len(digits) == 1:
        number = digits[0]
        tens = 0
        units = digits[0]
    elif len(digits) == 2:
        tens, units = digits
        number = tens * 10 + units
    else:
        raise ImageValidationError("Se requiere uno o dos dígitos.")
    return {
        "number": number,
        "tens": tens,
        "units": units,
        "digits": digits,
        "confidences": confidences,
        "confidence": float(np.mean(confidences)),
        "confidence_general": float(np.mean(confidences)),
        "probabilities": probabilities,
        "inputs": inputs,
    }
