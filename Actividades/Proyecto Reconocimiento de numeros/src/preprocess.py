"""Utilities to normalize, detect and segment handwritten digits."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class ImageValidationError(ValueError):
    """Raised when an input image cannot be interpreted as handwritten digits."""


def load_grayscale(image: Image.Image | str | Path) -> Image.Image:
    """Load an image and convert it to grayscale."""
    try:
        source = Image.open(image) if isinstance(image, (str, Path)) else image
        if source is None:
            raise ImageValidationError("No se recibio ninguna imagen.")
        grayscale = ImageOps.grayscale(source)
        if grayscale.width < 2 or grayscale.height < 2:
            raise ImageValidationError("La imagen es demasiado pequena.")
        return grayscale
    except (OSError, ValueError) as exc:
        raise ImageValidationError(f"Imagen invalida: {exc}") from exc


def _foreground(gray: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return locally corrected foreground intensity and a mask."""
    pixels = np.asarray(gray, dtype=np.uint8)
    border = np.concatenate((pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]))
    polarity_background = float(np.median(border))
    radius = max(3, min(gray.size) // 20)
    blurred = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float32,
    )
    pixels_float = pixels.astype(np.float32)
    if polarity_background >= 128:
        intensity = blurred - pixels_float
    elif float(np.percentile(pixels_float, 95)) - polarity_background >= 80:
        # Already-normalized white strokes on a dark background should keep
        # their filled intensity instead of becoming edge-only differences.
        intensity = pixels_float
    else:
        intensity = pixels_float - blurred
    intensity = np.clip(intensity, 0, 255)
    positive = intensity[intensity > 0]
    threshold = max(
        10.0,
        float(np.percentile(positive, 70)) * 0.45 if positive.size else 10.0,
    )
    mask = intensity > threshold
    mask = _clean_mask(mask)
    mask = _remove_long_background_lines(mask)
    return intensity, _morphology(mask)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove isolated noise while preserving thin handwritten strokes."""
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if min(image.size) < 5:
        return mask
    # Median filtering removes isolated specks without eroding a one-pixel
    # handwritten stroke as a binary opening would.
    cleaned = image.filter(ImageFilter.MedianFilter(3))
    return np.asarray(cleaned, dtype=np.uint8) > 0


def _morphology(mask: np.ndarray, *, thicken: bool = False) -> np.ndarray:
    """Apply conservative closing and optional adaptive thickening."""
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    # Closing = dilation followed by erosion. It joins short interruptions.
    closed = image.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    result = np.asarray(closed, dtype=np.uint8) > 0
    if thicken:
        active_ratio = float(result.mean())
        if active_ratio < 0.08:
            result = np.asarray(
                Image.fromarray((result.astype(np.uint8) * 255), mode="L").filter(
                    ImageFilter.MaxFilter(3)
                ),
                dtype=np.uint8,
            ) > 0
    return result


def _remove_long_background_lines(mask: np.ndarray) -> np.ndarray:
    """Remove nearly full-width or full-height background lines."""
    cleaned = mask.copy()
    if cleaned.ndim != 2:
        return cleaned
    row_limit = max(8, int(cleaned.shape[1] * 0.90))
    column_limit = max(8, int(cleaned.shape[0] * 0.90))
    rows = _thin_full_lines(cleaned.sum(axis=1) >= row_limit)
    columns = _thin_full_lines(cleaned.sum(axis=0) >= column_limit)
    cleaned[rows, :] = False
    cleaned[:, columns] = False
    return cleaned


def _thin_full_lines(values: np.ndarray, max_thickness: int = 2) -> np.ndarray:
    """Keep only isolated full-length lines, not thick digit strokes."""
    result = np.zeros_like(values, dtype=bool)
    for start, end in _split_columns(values):
        if end - start <= max_thickness:
            result[start:end] = True
    return result


def _mask_to_digit(mask: np.ndarray) -> Image.Image:
    """Convert a foreground mask to a high-contrast white-on-black image."""
    cleaned = mask
    coordinates = np.argwhere(cleaned)
    if coordinates.size == 0:
        raise ImageValidationError("No se detectaron trazos despues de limpiar el fondo.")
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    return Image.fromarray(
        (cleaned[y_min:y_max, x_min:x_max].astype(np.uint8) * 255),
        mode="L",
    )


def detect_strokes(image: Image.Image | str | Path) -> Tuple[Image.Image, np.ndarray]:
    """Crop the smallest rectangle containing foreground strokes."""
    grayscale = load_grayscale(image)
    intensity, mask = _foreground(grayscale)
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ImageValidationError("No se detectaron trazos.")
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    cropped = Image.fromarray(
        np.clip(intensity[y_min:y_max, x_min:x_max], 0, 255).astype(np.uint8),
        mode="L",
    )
    return cropped, mask[y_min:y_max, x_min:x_max]


def _split_columns(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous horizontal regions containing ink."""
    occupied = mask if mask.ndim == 1 else mask.any(axis=0)
    regions: List[Tuple[int, int]] = []
    start = None
    for index, value in enumerate(occupied):
        if value and start is None:
            start = index
        elif not value and start is not None:
            regions.append((start, index))
            start = None
    if start is not None:
        regions.append((start, len(occupied)))
    return regions


def _merge_small_gaps(
    regions: List[Tuple[int, int]], max_gap: int
) -> List[Tuple[int, int]]:
    if not regions:
        return []
    merged = [regions[0]]
    for start, end in regions[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def _close_column_gaps(occupied: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill short empty runs so broken strokes remain one digit."""
    closed = occupied.copy()
    for start, end in _split_columns(~occupied):
        if start > 0 and end < len(occupied) and end - start <= max_gap:
            closed[start:end] = True
    return closed


def _horizontal_erode(mask: np.ndarray, size: int = 3) -> np.ndarray:
    """Erode only across columns to expose a thin bridge between digits."""
    if size < 3 or size % 2 == 0:
        raise ValueError("size debe ser impar y mayor o igual que 3.")
    radius = size // 2
    padded = np.pad(mask, ((0, 0), (radius, radius)), constant_values=False)
    windows = [
        padded[:, offset : offset + mask.shape[1]]
        for offset in range(size)
    ]
    return np.logical_and.reduce(windows)


def _valley_boundary(mask: np.ndarray, minimum_width: int) -> int | None:
    """Find a low-density vertical valley suitable for splitting two digits."""
    profile = mask.sum(axis=0).astype(np.float32)
    if profile.size < minimum_width * 2:
        return None
    kernel = np.ones(5, dtype=np.float32) / 5
    smooth = np.convolve(profile, kernel, mode="same")
    first = max(1, int(profile.size * 0.25))
    last = min(profile.size - 1, int(profile.size * 0.75))
    if last <= first:
        return None
    boundary = first + int(np.argmin(smooth[first:last]))
    left_peak = float(np.max(smooth[first:boundary])) if boundary > first else 0
    right_peak = float(np.max(smooth[boundary:last])) if last > boundary else 0
    valley = float(smooth[boundary])
    if min(left_peak, right_peak) <= 0 or valley > min(left_peak, right_peak) * 0.78:
        return None
    return boundary


def _extract_mask_region(
    mask: np.ndarray, start: int, end: int
) -> Image.Image:
    """Extract one digit directly from the cleaned binary mask."""
    part = mask[:, start:end]
    coordinates = np.argwhere(part)
    if coordinates.size == 0:
        raise ImageValidationError("No se detectaron trazos en la region.")
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    return Image.fromarray(
        (part[y_min:y_max, x_min:x_max].astype(np.uint8) * 255),
        mode="L",
    )


def _split_at_boundary(mask: np.ndarray, boundary: int) -> List[Image.Image]:
    """Split a cropped number at a chosen vertical boundary."""
    result: List[Image.Image] = []
    for start, end in ((0, boundary), (boundary, mask.shape[1])):
        if end - start < 2:
            raise ImageValidationError("La separacion entre digitos no es valida.")
        result.append(_extract_mask_region(mask, start, end))
    return result


def split_two_digits(image: Image.Image | str | Path) -> List[Image.Image]:
    """Split an image into two vertically adjacent digit crops."""
    cropped, mask = detect_strokes(image)
    # Keep the original projection here. Closing gaps is useful inside one
    # digit, but it can erase the intentional space between two digits.
    occupied = mask.any(axis=0)
    columns = np.flatnonzero(occupied)
    if columns.size < 2:
        raise ImageValidationError("No se detectaron dos digitos separados.")
    gaps = np.diff(columns)
    split_at = int(np.argmax(gaps))
    largest_gap = int(gaps[split_at])
    boundary = None
    if largest_gap >= max(3, cropped.width // 30):
        boundary = int((columns[split_at] + columns[split_at + 1]) / 2)
    if boundary is None:
        boundary = _valley_boundary(_horizontal_erode(mask, 3), cropped.width // 8)
    if boundary is None and cropped.width >= cropped.height * 1.45:
        # When the digits touch continuously there may be no empty valley.
        # A centered cut is safer than returning only the left digit.
        boundary = cropped.width // 2
    if boundary is None:
        raise ImageValidationError("No se detectaron dos regiones de digitos.")
    return _split_at_boundary(mask, boundary)


def segment_digits(
    image: Image.Image | str | Path, expected_digits: int | None = None
) -> List[Image.Image]:
    """Detect one or two digits and return each cropped grayscale image."""
    cropped, mask = detect_strokes(image)
    raw_occupied = mask.any(axis=0)
    raw_columns = np.flatnonzero(raw_occupied)

    # Detect a clear inter-digit valley before filling small gaps inside
    # strokes. This is the important distinction for inputs such as "45".
    if expected_digits in (None, 2) and raw_columns.size > 1:
        raw_gaps = np.diff(raw_columns)
        raw_split = int(np.argmax(raw_gaps))
        if int(raw_gaps[raw_split]) >= max(3, cropped.width // 30):
            return split_two_digits(image)
        if expected_digits == 2:
            eroded_boundary = _valley_boundary(
                _horizontal_erode(mask, 3),
                cropped.width // 8,
            )
            if eroded_boundary is not None:
                return _split_at_boundary(mask, eroded_boundary)

    gap_size = max(2, cropped.width // 25)
    occupied = _close_column_gaps(mask.any(axis=0), max_gap=gap_size)
    regions = _merge_small_gaps(_split_columns(occupied), max_gap=gap_size)
    regions = [(start, end) for start, end in regions if end - start >= 2]

    if expected_digits is not None and len(regions) != expected_digits:
        if expected_digits == 2:
            return split_two_digits(image)
        raise ImageValidationError(
            f"Se esperaban {expected_digits} digitos, pero se detectaron {len(regions)}."
        )
    if len(regions) > 2:
        return split_two_digits(image)

    # If two digits touch or the projection has one connected region, use the
    # widest empty vertical gap as a split only when it is clearly visible.
    if expected_digits is None and len(regions) == 1:
        columns = np.flatnonzero(occupied)
        if columns.size > 1:
            gaps = np.diff(columns)
            split_at = int(np.argmax(gaps))
            if int(gaps[split_at]) >= max(3, cropped.width // 12):
                return split_two_digits(image)
        if cropped.width > cropped.height * 1.35:
            boundary = _valley_boundary(_horizontal_erode(mask, 3), cropped.width // 8)
            if boundary is not None:
                return _split_at_boundary(mask, boundary)
            if cropped.width >= cropped.height * 1.45:
                return split_two_digits(image)

    if not regions:
        raise ImageValidationError("No se pudieron segmentar digitos.")
    result: List[Image.Image] = []
    for start, end in regions:
        result.append(_extract_mask_region(mask, start, end))
    return result


def preprocess_single_digit(image: Image.Image | str | Path) -> np.ndarray:
    """Center one digit on a 28x28 canvas and return a model input vector."""
    source = load_grayscale(image)
    _, mask = _foreground(source)
    pixels = np.asarray(source, dtype=np.uint8)
    if not mask.any() and float(np.percentile(pixels, 5)) > 220:
        # A segmented crop can contain only foreground pixels. Add a dark
        # border so the foreground detector can recover its shape.
        source = ImageOps.expand(source, border=2, fill=0)
        _, mask = _foreground(source)
    mask = _morphology(mask, thicken=True)
    digit = _mask_to_digit(mask)
    scale = 20 / max(digit.width, digit.height)
    resized = digit.resize(
        (max(1, round(digit.width * scale)), max(1, round(digit.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(resized, ((28 - resized.width) // 2, (28 - resized.height) // 2))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).reshape(1, 784)


def normalized_digit_image(
    image: Image.Image | str | Path, scale: int = 8
) -> Image.Image:
    """Return the exact 28x28 model input enlarged for visual inspection."""
    digit = preprocess_single_digit(image).reshape(28, 28)
    result = Image.fromarray(np.clip(digit * 255, 0, 255).astype(np.uint8), mode="L")
    return result.resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def preprocessing_preview(
    image: Image.Image | str | Path,
) -> Tuple[Image.Image, Image.Image, Image.Image, List[Image.Image]]:
    """Return grayscale, binary, cropped and normalized debug images."""
    grayscale = load_grayscale(image)
    _, mask = _foreground(grayscale)
    binary = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    cropped = _mask_to_digit(mask)
    parts = segment_digits(image)
    normalized = [normalized_digit_image(part) for part in parts]
    return grayscale, binary, cropped, normalized


def digit_to_mnist(image: Image.Image | str | Path) -> np.ndarray:
    return preprocess_single_digit(image)


def prepare_for_model(image: Image.Image | str | Path) -> np.ndarray:
    return preprocess_single_digit(image)


def prepare_input(
    image: Image.Image | str | Path, expected_digits: int | None = None
) -> List[np.ndarray]:
    return [digit_to_mnist(part) for part in segment_digits(image, expected_digits)]
