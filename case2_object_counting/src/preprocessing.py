"""
preprocessing.py — Caso 2: Conteo Automático de Objetos

Funciones de preparación de imágenes previas a la binarización:
    - Generación de imágenes SINTÉTICAS con conteo real (ground truth)
      conocido, para poder evaluar cuantitativamente la precisión del
      pipeline de conteo (necesarias porque no siempre se dispone de un
      dataset anotado).
    - Conversión a escala de grises.
    - Reducción de ruido (denoising) previa a la binarización.
    - Realce de contraste.

Todas las funciones:
    - Reciben y devuelven imágenes en formato NumPy array (uint8).
    - No modifican la imagen de entrada in-place (devuelven una copia).

Autor: Portafolio Python for Research — Módulo 2
"""

from __future__ import annotations

import numpy as np
import cv2


# ----------------------------------------------------------------------
# Generación de datos sintéticos (con ground truth conocido)
# ----------------------------------------------------------------------

def generate_synthetic_particles(
    image_size: tuple[int, int] = (400, 400),
    num_particles: int = 40,
    radius_range: tuple[int, int] = (8, 18),
    allow_overlap: bool = True,
    noise_sigma: float = 8.0,
    seed: int | None = 42,
) -> tuple[np.ndarray, dict]:
    """Genera una imagen sintética de "partículas" circulares con conteo real conocido.

    Simula un escenario típico de conteo automático (colonias, células,
    partículas, monedas, etc.): círculos claros sobre fondo oscuro,
    algunos de ellos superpuestos/tocándose, lo cual es intencional para
    poder demostrar la diferencia entre un conteo ingenuo por contornos
    (subestima objetos que se tocan) y un conteo basado en Watershed
    (separa correctamente objetos adyacentes).

    Parameters
    ----------
    image_size : tuple[int, int], default=(400, 400)
        Tamaño (alto, ancho) del lienzo generado.
    num_particles : int, default=40
        Número de partículas (círculos) a dibujar. Este es el "ground
        truth" contra el que se comparan los métodos de conteo.
    radius_range : tuple[int, int], default=(8, 18)
        Rango (mínimo, máximo) del radio de cada partícula en píxeles.
    allow_overlap : bool, default=True
        Si es True, se permiten solapamientos entre partículas (más
        realista y más exigente para el pipeline de conteo). Si es
        False, se reintenta el posicionamiento hasta evitar solapes.
    noise_sigma : float, default=8.0
        Desviación estándar del ruido Gaussiano añadido para simular
        una imagen realista (sensor/adquisición), no un dibujo perfecto.
    seed : int, optional, default=42
        Semilla para reproducibilidad.

    Returns
    -------
    tuple[np.ndarray, dict]
        - Imagen sintética (uint8, escala de grises), fondo oscuro con
          partículas claras.
        - Diccionario con metadatos de ground truth: ``{"count": int,
          "centers": list[tuple[int, int]], "radii": list[int]}``.
    """
    rng = np.random.default_rng(seed)
    h, w = image_size
    canvas = np.full((h, w), 30, dtype=np.uint8)  # fondo oscuro no uniforme

    centers: list[tuple[int, int]] = []
    radii: list[int] = []

    margin = radius_range[1] + 2
    max_attempts_per_particle = 50

    for _ in range(num_particles):
        placed = False
        for _ in range(max_attempts_per_particle):
            cx = int(rng.integers(margin, w - margin))
            cy = int(rng.integers(margin, h - margin))
            r = int(rng.integers(radius_range[0], radius_range[1] + 1))

            if not allow_overlap:
                overlaps = any(
                    (cx - ex) ** 2 + (cy - ey) ** 2 < (r + er) ** 2
                    for (ex, ey), er in zip(centers, radii)
                )
                if overlaps:
                    continue

            intensity = int(rng.integers(170, 240))
            cv2.circle(canvas, (cx, cy), r, intensity, thickness=-1)
            centers.append((cx, cy))
            radii.append(r)
            placed = True
            break

        if not placed:
            # No se encontró posición libre tras varios intentos: se omite
            # esta partícula (solo relevante cuando allow_overlap=False).
            continue

    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, canvas.shape)
        canvas = np.clip(canvas.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    ground_truth = {"count": len(centers), "centers": centers, "radii": radii}
    return canvas, ground_truth


# ----------------------------------------------------------------------
# Conversión de espacio de color
# ----------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen a escala de grises si no lo está ya.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8), BGR (color) o escala de grises.

    Returns
    -------
    np.ndarray
        Imagen en escala de grises (uint8), 2D.
    """
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ----------------------------------------------------------------------
# Reducción de ruido
# ----------------------------------------------------------------------

def denoise(image: np.ndarray, method: str = "gaussian", **kwargs) -> np.ndarray:
    """Reduce ruido en la imagen antes de la binarización.

    Un paso de denoising previo evita que la binarización (sensible a
    variaciones locales de intensidad) genere ruido tipo "sal y pimienta"
    en la máscara binaria resultante.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8), escala de grises o color.
    method : str, default="gaussian"
        Método de filtrado: ``"gaussian"``, ``"median"`` o ``"bilateral"``.
    **kwargs
        Parámetros adicionales pasados al filtro correspondiente
        (p. ej. ``kernel_size`` para gaussian/median, o ``d``,
        ``sigma_color``, ``sigma_space`` para bilateral).

    Returns
    -------
    np.ndarray
        Imagen filtrada (uint8), mismo shape que la entrada.
    """
    if method == "gaussian":
        kernel_size = kwargs.get("kernel_size", (5, 5))
        return cv2.GaussianBlur(image, kernel_size, kwargs.get("sigma", 0.0))
    if method == "median":
        kernel_size = kwargs.get("kernel_size", 5)
        return cv2.medianBlur(image, kernel_size)
    if method == "bilateral":
        return cv2.bilateralFilter(
            image,
            kwargs.get("d", 9),
            kwargs.get("sigma_color", 75.0),
            kwargs.get("sigma_space", 75.0),
        )
    raise ValueError(f"Método de denoising desconocido: '{method}'. "
                      f"Opciones válidas: 'gaussian', 'median', 'bilateral'.")


# ----------------------------------------------------------------------
# Realce de contraste
# ----------------------------------------------------------------------

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0,
                      tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Aplica CLAHE para mejorar el contraste local antes de la binarización.

    Útil cuando la iluminación no es uniforme sobre la imagen, escenario
    en el que un único umbral global (p. ej. Otsu) puede fallar en zonas
    con distinto nivel de brillo.

    Parameters
    ----------
    image : np.ndarray
        Imagen en escala de grises (uint8).
    clip_limit : float, default=2.0
        Límite de contraste por tile.
    tile_grid_size : tuple[int, int], default=(8, 8)
        Número de tiles (filas, columnas).

    Returns
    -------
    np.ndarray
        Imagen con contraste realzado (uint8).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def prepare_for_segmentation(image: np.ndarray, denoise_method: str = "gaussian",
                              apply_clahe: bool = False, **denoise_kwargs) -> np.ndarray:
    """Pipeline corto de preprocesamiento: escala de grises + denoising (+ CLAHE opcional).

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8), color o escala de grises.
    denoise_method : str, default="gaussian"
        Método de denoising a usar (ver :func:`denoise`).
    apply_clahe : bool, default=False
        Si es True, aplica CLAHE tras el denoising.
    **denoise_kwargs
        Parámetros adicionales para :func:`denoise`.

    Returns
    -------
    np.ndarray
        Imagen en escala de grises, lista para binarizar (uint8).
    """
    gray = to_grayscale(image)
    filtered = denoise(gray, method=denoise_method, **denoise_kwargs)
    if apply_clahe:
        filtered = enhance_contrast(filtered)
    return filtered
