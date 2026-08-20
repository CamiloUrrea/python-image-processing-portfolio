from __future__ import annotations

import numpy as np
import cv2

import detection as det
import ocr as ocr_mod


def build_default_ocr_reader(languages: tuple[str, ...] = ("en",), gpu: bool = False) -> ocr_mod.OCRReader:
    return ocr_mod.EasyOCRReader(languages=languages, gpu=gpu)


def process_frame(frame: np.ndarray, reader: ocr_mod.OCRReader,
                   min_confidence: float = 0.2, upscale: float = 2.0) -> dict:
    candidates = det.locate_plate_candidates(frame)

    if not candidates:
        return {
            "success": False,
            "bbox": None,
            "text": None,
            "confidence": 0.0,
            "roi": None,
            "preprocessed": None,
            "candidates": [],
        }

    best = candidates[0]
    roi = det.extract_roi(frame, best["bbox"])
    preprocessed = ocr_mod.preprocess_plate_roi(roi, upscale=upscale)
    reading = ocr_mod.read_plate_text(reader, preprocessed["ocr_input"], min_confidence=min_confidence)

    return {
        "success": reading["text"] is not None,
        "bbox": best["bbox"],
        "text": reading["text"],
        "confidence": reading["confidence"],
        "roi": roi,
        "preprocessed": preprocessed,
        "candidates": candidates,
    }


def process_video_frames(frames: list[np.ndarray], reader: ocr_mod.OCRReader,
                          min_confidence: float = 0.2, upscale: float = 2.0) -> list[dict]:
    return [
        process_frame(frame, reader, min_confidence=min_confidence, upscale=upscale)
        for frame in frames
    ]


def draw_plate_result(frame: np.ndarray, result: dict,
                       box_color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    vis = frame.copy()
    if result["bbox"] is None:
        return vis

    x, y, w, h = result["bbox"]
    cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 2)

    label = result["text"] if result["text"] else "N/D"
    caption = f"{label} ({result['confidence']:.2f})"
    (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y0 = max(0, y - text_h - 10)
    cv2.rectangle(vis, (x, label_y0), (x + text_w + 6, y), box_color, -1)
    cv2.putText(vis, caption, (x + 3, max(text_h, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    return vis


def generate_synthetic_vehicle_frame(image_size: tuple[int, int] = (480, 640),
                                      plate_text: str = "ABC1234",
                                      seed: int | None = None) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    height, width = image_size

    background_level = int(rng.integers(90, 150))
    frame = np.full((height, width, 3), background_level, dtype=np.uint8)

    car_color = tuple(int(c) for c in rng.integers(40, 200, size=3))
    car_x0, car_y0 = int(width * 0.2), int(height * 0.25)
    car_x1, car_y1 = int(width * 0.8), int(height * 0.75)
    cv2.rectangle(frame, (car_x0, car_y0), (car_x1, car_y1), car_color, -1)

    plate_w, plate_h = int(width * 0.22), int(height * 0.09)
    plate_x0 = (car_x0 + car_x1) // 2 - plate_w // 2
    plate_y0 = int(car_y1 - plate_h * 1.6)
    plate_x1, plate_y1 = plate_x0 + plate_w, plate_y0 + plate_h

    cv2.rectangle(frame, (plate_x0, plate_y0), (plate_x1, plate_y1), (255, 255, 255), -1)
    cv2.rectangle(frame, (plate_x0, plate_y0), (plate_x1, plate_y1), (0, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = plate_h / 45.0
    thickness = max(1, int(plate_h / 20))
    (text_w, text_h), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    text_x = plate_x0 + (plate_w - text_w) // 2
    text_y = plate_y0 + (plate_h + text_h) // 2
    cv2.putText(frame, plate_text, (text_x, text_y), font, font_scale,
                (0, 0, 0), thickness, cv2.LINE_AA)

    noise = rng.normal(0, 6, frame.shape)
    frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    ground_truth = {
        "plate_text": plate_text,
        "bbox": (plate_x0, plate_y0, plate_w, plate_h),
    }
    return frame, ground_truth
