"""Gradio interface for handwritten number recognition."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model, predict_number
from src.preprocess import ImageValidationError, preprocessing_preview, segment_digits


MODEL_PATH = PROJECT_ROOT / "models" / "mnist_digits.keras"
MODEL = None


def get_model() -> Any:
    """Load the model lazily and reuse it for subsequent predictions."""
    global MODEL
    if MODEL is None:
        MODEL = load_model(MODEL_PATH)
    return MODEL


def recognize(
    image: Image.Image | None,
) -> tuple[
    str,
    str,
    Image.Image,
    Image.Image,
    Image.Image,
    Image.Image,
    list[list[object]],
]:
    """Recognize one or two digits and return preprocessing diagnostics."""
    if image is None:
        raise gr.Error("Carga una imagen PNG, JPG o JPEG antes de reconocerla.")
    try:
        grayscale, binary, cropped, _ = preprocessing_preview(image)
        parts = segment_digits(image)
        result = predict_number(image, get_model())

        processed = Image.new("L", (28 * len(parts), 28), color=0)
        for index, vector in enumerate(result["inputs"]):
            digit = (vector.reshape(28, 28) * 255).astype("uint8")
            processed.paste(Image.fromarray(digit), (index * 28, 0))
        processed = processed.resize(
            (processed.width * 8, processed.height * 8),
            Image.Resampling.NEAREST,
        )

        probabilities = [
            [f"Digito {index + 1}: clase {label}", f"{probability:.6f}"]
            for index, values in enumerate(result["probabilities"])
            for label, probability in enumerate(values)
        ]
        return (
            str(result["number"]),
            f"{result['confidence_general']:.2%}",
            grayscale,
            binary.resize((binary.width * 3, binary.height * 3), Image.Resampling.NEAREST),
            cropped.resize((cropped.width * 3, cropped.height * 3), Image.Resampling.NEAREST),
            processed,
            probabilities,
        )
    except (ImageValidationError, FileNotFoundError, OSError) as error:
        raise gr.Error(f"No se pudo procesar la imagen: {error}") from error


THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
)


with gr.Blocks(title="Reconocimiento de numeros del 0 al 99") as demo:
    gr.Markdown("# Reconocimiento de numeros del 0 al 99")
    gr.Markdown(
        "Carga o arrastra una imagen con uno o dos digitos manuscritos. "
        "Las vistas de preprocesamiento se muestran ampliadas para inspeccionar "
        "lo que recibe el modelo."
    )
    image_input = gr.Image(
        type="pil",
        sources=["upload", "clipboard"],
        label="Imagen de entrada",
    )
    recognize_button = gr.Button("Reconocer numero", variant="primary")
    with gr.Row():
        number_output = gr.Textbox(label="Numero reconocido")
        confidence_output = gr.Textbox(label="Confianza")
    with gr.Row():
        grayscale_output = gr.Image(type="pil", label="Escala de grises")
        binary_output = gr.Image(type="pil", label="Blanco y negro")
        cropped_output = gr.Image(type="pil", label="Trazos recortados")
    processed_output = gr.Image(
        type="pil",
        label="Entrada final del modelo (ampliada)",
    )
    probabilities_output = gr.Dataframe(
        headers=["Clase", "Probabilidad"],
        datatype=["str", "str"],
        label="Probabilidades por digito/clase",
        interactive=False,
    )
    recognize_button.click(
        recognize,
        inputs=image_input,
        outputs=[
            number_output,
            confidence_output,
            grayscale_output,
            binary_output,
            cropped_output,
            processed_output,
            probabilities_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(theme=THEME)
