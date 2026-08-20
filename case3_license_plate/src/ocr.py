from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np
import cv2

_ALPHANUMERIC_PATTERN = re.compile(r"[^A-Z0-9]")


class OCRReader(ABC):
    @abstractmethod
    def read(self, image: np.ndarray) -> list[dict]:
        raise NotImplementedError


class EasyOCRReader(OCRReader):
    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = False):
        import easyocr
        self._reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)

    def read(self, image: np.ndarray) -> list[dict]:
        raw_results = self._reader.readtext(image)
        return [
            {"text": text, "confidence": float(confidence), "bbox": bbox}
            for bbox, text, confidence in raw_results
        ]


def resize_for_ocr(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    height, width = image.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def denoise_roi(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.medianBlur(gray, kernel_size)


def binarize_roi(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def preprocess_plate_roi(roi: np.ndarray, upscale: float = 2.0) -> dict:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    resized = resize_for_ocr(gray, scale=upscale)
    denoised = denoise_roi(resized)
    binary = binarize_roi(denoised)

    return {
        "gray": gray,
        "resized": resized,
        "denoised": denoised,
        "binary": binary,
        "ocr_input": denoised,
    }


def clean_plate_text(text: str) -> str:
    return _ALPHANUMERIC_PATTERN.sub("", text.upper())


def select_best_reading(ocr_results: list[dict], min_confidence: float = 0.0) -> dict | None:
    candidates = []
    for result in ocr_results:
        cleaned = clean_plate_text(result["text"])
        if not cleaned or result["confidence"] < min_confidence:
            continue
        candidates.append({"text": cleaned, "confidence": result["confidence"]})

    if not candidates:
        return None

    return max(candidates, key=lambda c: c["confidence"])


def read_plate_text(reader: OCRReader, roi: np.ndarray, min_confidence: float = 0.2) -> dict:
    raw_results = reader.read(roi)
    best = select_best_reading(raw_results, min_confidence=min_confidence)

    return {
        "text": best["text"] if best else None,
        "confidence": best["confidence"] if best else 0.0,
        "raw_results": raw_results,
    }
