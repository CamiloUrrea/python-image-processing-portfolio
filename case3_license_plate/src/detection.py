from __future__ import annotations

import numpy as np
import cv2


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_edge_preserving(gray: np.ndarray, d: int = 11,
                             sigma_color: float = 17.0,
                             sigma_space: float = 17.0) -> np.ndarray:
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def sobel_gradient_magnitude(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    magnitude = np.absolute(grad_x)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def binarize_gradient(gradient: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def close_plate_regions(mask: np.ndarray, kernel_size: tuple[int, int] = (17, 3)) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def refine_candidate_mask(mask: np.ndarray, kernel_size: int = 3,
                           erode_iterations: int = 1,
                           dilate_iterations: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    result = cv2.erode(mask, kernel, iterations=erode_iterations)
    return cv2.dilate(result, kernel, iterations=dilate_iterations)


def find_plate_candidates(mask: np.ndarray,
                           min_aspect_ratio: float = 2.0,
                           max_aspect_ratio: float = 6.5,
                           min_area: int = 300,
                           max_area_ratio: float = 0.25,
                           image_area: int | None = None) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if h == 0 or area < min_area:
            continue
        if image_area is not None and area > max_area_ratio * image_area:
            continue

        aspect_ratio = w / float(h)
        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            continue

        candidates.append({
            "bbox": (x, y, w, h),
            "area": area,
            "aspect_ratio": aspect_ratio,
            "contour": contour,
        })

    return sorted(candidates, key=lambda c: c["area"], reverse=True)


def locate_plate_candidates(image: np.ndarray,
                             gradient_kernel_size: tuple[int, int] = (17, 3),
                             refine_kernel_size: int = 3,
                             min_aspect_ratio: float = 2.0,
                             max_aspect_ratio: float = 6.5,
                             min_area: int = 300,
                             max_area_ratio: float = 0.25) -> list[dict]:
    gray = to_grayscale(image)
    denoised = denoise_edge_preserving(gray)
    gradient = sobel_gradient_magnitude(denoised)
    binary = binarize_gradient(gradient)
    closed = close_plate_regions(binary, gradient_kernel_size)
    refined = refine_candidate_mask(closed, refine_kernel_size)

    image_area = gray.shape[0] * gray.shape[1]
    return find_plate_candidates(
        refined,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        image_area=image_area,
    )


def extract_roi(image: np.ndarray, bbox: tuple[int, int, int, int],
                 padding: int = 4) -> np.ndarray:
    x, y, w, h = bbox
    height, width = image.shape[:2]
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, width)
    y1 = min(y + h + padding, height)
    return image[y0:y1, x0:x1]
