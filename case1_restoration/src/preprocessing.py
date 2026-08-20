"""
preprocessing.py — Caso 1: Restauración y Mejora de Fotografías Degradadas

Funciones para SIMULAR degradaciones controladas sobre imágenes limpias.
Esto permite construir pares (imagen_original, imagen_degradada) con
ground truth conocido, necesarios para evaluar cuantitativamente
(PSNR/SSIM) la efectividad del pipeline de restauración.

Todas las funciones:
    - Reciben y devuelven imágenes en formato NumPy array (uint8, BGR u OpenCV-compatible).
    - No modifican la imagen de entrada in-place (devuelven una copia).
    - Están documentadas con docstrings estilo NumPy/Google.

Autor: Portafolio Python for Research — Módulo 1
"""

from __future__ import annotations

import numpy as np
import cv2


# ----------------------------------------------------------------------
# Ruido
# ----------------------------------------------------------------------

def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0,
                        seed: int | None = None) -> np.ndarray:
    """Añade ruido Gaussiano (aditivo) a la imagen.

    Simula ruido de sensor / baja iluminación, típico en fotografías
    tomadas con cámaras de bajo costo o en condiciones de poca luz.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8), escala de grises o color (H, W[, C]).
    mean : float, default=0.0
        Media de la distribución Gaussiana del ruido.
    sigma : float, default=25.0
        Desviación estándar del ruido. A mayor sigma, mayor degradación.
    seed : int, optional
        Semilla para reproducibilidad.

    Returns
    -------
    np.ndarray
        Imagen degradada (uint8), mismo shape que la entrada.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.05,
                           salt_vs_pepper: float = 0.5,
                           seed: int | None = None) -> np.ndarray:
    """Añade ruido impulsivo "sal y pimienta" a la imagen.

    Simula errores de transmisión / píxeles defectuosos: reemplaza una
    fracción de píxeles aleatorios por blanco puro (sal) o negro puro
    (pimienta).

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8).
    amount : float, default=0.05
        Proporción de píxeles totales afectados (0.0 - 1.0).
    salt_vs_pepper : float, default=0.5
        Proporción de píxeles "sal" (blancos) respecto al total de
        píxeles afectados. El resto son "pimienta" (negros).
    seed : int, optional
        Semilla para reproducibilidad.

    Returns
    -------
    np.ndarray
        Imagen degradada (uint8), mismo shape que la entrada.
    """
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    h, w = image.shape[:2]
    num_pixels = int(amount * h * w)

    # Sal (blanco)
    num_salt = int(num_pixels * salt_vs_pepper)
    coords = (rng.integers(0, h, num_salt), rng.integers(0, w, num_salt))
    noisy[coords] = 255

    # Pimienta (negro)
    num_pepper = num_pixels - num_salt
    coords = (rng.integers(0, h, num_pepper), rng.integers(0, w, num_pepper))
    noisy[coords] = 0

    return noisy


# ----------------------------------------------------------------------
# Desenfoque (blur)
# ----------------------------------------------------------------------

def add_gaussian_blur(image: np.ndarray, kernel_size: tuple[int, int] = (9, 9),
                       sigma: float = 0.0) -> np.ndarray:
    """Aplica desenfoque Gaussiano (simula desenfoque de foco/lente).

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8).
    kernel_size : tuple[int, int], default=(9, 9)
        Tamaño del kernel (debe contener valores impares y positivos).
    sigma : float, default=0.0
        Desviación estándar en X e Y. Si es 0, se calcula automáticamente
        a partir del tamaño del kernel (comportamiento de OpenCV).

    Returns
    -------
    np.ndarray
        Imagen degradada (uint8).
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def add_motion_blur(image: np.ndarray, size: int = 15, angle: float = 0.0) -> np.ndarray:
    """Simula desenfoque de movimiento (motion blur) con un kernel lineal.

    Útil para simular fotografías tomadas con cámara/objeto en movimiento.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8).
    size : int, default=15
        Longitud del kernel de movimiento en píxeles.
    angle : float, default=0.0
        Ángulo del movimiento en grados (0 = horizontal).

    Returns
    -------
    np.ndarray
        Imagen degradada (uint8).
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[(size - 1) // 2, :] = 1.0
    center = (size / 2 - 0.5, size / 2 - 0.5)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot_matrix, (size, size))
    kernel /= kernel.sum() if kernel.sum() != 0 else 1.0
    return cv2.filter2D(image, -1, kernel)


# ----------------------------------------------------------------------
# Contraste / brillo / resolución / compresión
# ----------------------------------------------------------------------

def reduce_contrast(image: np.ndarray, factor: float = 0.5,
                     brightness_shift: int = 0) -> np.ndarray:
    """Reduce el contraste de la imagen (y opcionalmente desplaza el brillo).

    Simula fotografías "planas" (baja dinámica), típicas de escaneos
    antiguos o fotografías subexpuestas/sobreexpuestas.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8).
    factor : float, default=0.5
        Factor de escala del contraste. Valores < 1.0 reducen el contraste;
        valores > 1.0 lo aumentan.
    brightness_shift : int, default=0
        Desplazamiento aditivo de brillo (-255 a 255).

    Returns
    -------
    np.ndarray
        Imagen degradada (uint8).
    """
    img_f = image.astype(np.float64)
    mean_val = img_f.mean()
    degraded = (img_f - mean_val) * factor + mean_val + brightness_shift
    return np.clip(degraded, 0, 255).astype(np.uint8)


def reduce_resolution(image: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """Simula baja resolución: reduce y luego re-escala la imagen a su tamaño original.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8).
    scale : float, default=0.25
        Factor de reducción (0 < scale < 1). Valores más pequeños generan
        mayor pérdida de detalle.

    Returns
    -------
    np.ndarray
        Imagen degradada (uint8), mismo shape que la entrada original.
    """
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def add_jpeg_artifacts(image: np.ndarray, quality: int = 15) -> np.ndarray:
    """Simula artefactos de compresión JPEG agresiva (bloques/ringing).

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8).
    quality : int, default=15
        Calidad JPEG (1-100). Valores bajos generan más artefactos.

    Returns
    -------
    np.ndarray
        Imagen degradada (uint8).
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded = cv2.imencode(".jpg", image, encode_param)
    if not success:
        raise RuntimeError("No se pudo codificar la imagen en JPEG.")
    flag = cv2.IMREAD_COLOR if image.ndim == 3 else cv2.IMREAD_GRAYSCALE
    return cv2.imdecode(encoded, flag)


# ----------------------------------------------------------------------
# Pipeline combinado
# ----------------------------------------------------------------------

DEGRADATION_REGISTRY = {
    "gaussian_noise": add_gaussian_noise,
    "salt_pepper_noise": add_salt_pepper_noise,
    "gaussian_blur": add_gaussian_blur,
    "motion_blur": add_motion_blur,
    "low_contrast": reduce_contrast,
    "low_resolution": reduce_resolution,
    "jpeg_artifacts": add_jpeg_artifacts,
}


def apply_degradation_pipeline(image: np.ndarray,
                                steps: list[tuple[str, dict]]) -> np.ndarray:
    """Aplica una secuencia de degradaciones en cadena sobre la imagen.

    Permite construir escenarios de degradación combinada y realista
    (p. ej. desenfoque + ruido + baja resolución), en lugar de evaluar
    una sola degradación aislada.

    Parameters
    ----------
    image : np.ndarray
        Imagen original (limpia), uint8.
    steps : list[tuple[str, dict]]
        Lista de pasos a aplicar en orden. Cada paso es una tupla
        ``(nombre_degradacion, kwargs)``, donde ``nombre_degradacion``
        debe existir en ``DEGRADATION_REGISTRY``.

    Returns
    -------
    np.ndarray
        Imagen degradada resultante (uint8).

    Examples
    --------
    >>> steps = [
    ...     ("gaussian_blur", {"kernel_size": (7, 7)}),
    ...     ("gaussian_noise", {"sigma": 15}),
    ... ]
    >>> degraded = apply_degradation_pipeline(clean_image, steps)
    """
    result = image.copy()
    for name, kwargs in steps:
        if name not in DEGRADATION_REGISTRY:
            raise ValueError(
                f"Degradación desconocida: '{name}'. "
                f"Opciones válidas: {list(DEGRADATION_REGISTRY.keys())}"
            )
        result = DEGRADATION_REGISTRY[name](result, **kwargs)
    return result