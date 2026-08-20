"""
evaluation.py — Caso 1: Restauración y Mejora de Fotografías Degradadas

Métricas cuantitativas de calidad de imagen (full-reference: requieren
la imagen original/limpia como ground truth) y utilidades para tabular
comparaciones entre múltiples versiones de una misma imagen
(original vs. degradada vs. restaurada).

Métricas implementadas:
    - MSE   (Mean Squared Error)
    - PSNR  (Peak Signal-to-Noise Ratio, en dB)
    - SSIM  (Structural Similarity Index Measure)

Autor: Portafolio Python for Research — Módulo 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.metrics import (
    peak_signal_noise_ratio as skimage_psnr,
    structural_similarity as skimage_ssim,
    mean_squared_error as skimage_mse,
)


def _validate_pair(original: np.ndarray, comparison: np.ndarray) -> None:
    """Valida que dos imágenes tengan el mismo shape antes de comparar."""
    if original.shape != comparison.shape:
        raise ValueError(
            f"Las imágenes deben tener el mismo shape para comparar. "
            f"original={original.shape}, comparison={comparison.shape}"
        )


def compute_mse(original: np.ndarray, comparison: np.ndarray) -> float:
    """Calcula el Error Cuadrático Medio (MSE) entre dos imágenes.

    Métrica base de diferencia píxel a píxel. Valores más bajos indican
    mayor similitud (0 = imágenes idénticas). No siempre correlaciona
    bien con la percepción visual humana de calidad.

    Parameters
    ----------
    original : np.ndarray
        Imagen de referencia (ground truth), uint8.
    comparison : np.ndarray
        Imagen a evaluar (degradada o restaurada), uint8, mismo shape.

    Returns
    -------
    float
        Valor de MSE (a menor valor, mayor similitud).
    """
    _validate_pair(original, comparison)
    return float(skimage_mse(original, comparison))


def compute_psnr(original: np.ndarray, comparison: np.ndarray,
                  data_range: int = 255) -> float:
    """Calcula el PSNR (Peak Signal-to-Noise Ratio) entre dos imágenes.

    Expresa, en decibeles (dB), la relación entre la máxima potencia
    posible de la señal y la potencia del error (MSE). Es la métrica
    estándar en restauración de imágenes: valores más altos indican
    mejor calidad / mayor fidelidad respecto al original.

    Referencia orientativa: >30 dB suele considerarse buena calidad;
    >40 dB, muy alta fidelidad (varía según el dominio de aplicación).

    Parameters
    ----------
    original : np.ndarray
        Imagen de referencia (ground truth), uint8.
    comparison : np.ndarray
        Imagen a evaluar, uint8, mismo shape que `original`.
    data_range : int, default=255
        Rango dinámico de los valores de píxel (255 para uint8).

    Returns
    -------
    float
        Valor de PSNR en dB. Puede ser `inf` si las imágenes son idénticas.
    """
    _validate_pair(original, comparison)
    return float(skimage_psnr(original, comparison, data_range=data_range))


def compute_ssim(original: np.ndarray, comparison: np.ndarray,
                  data_range: int = 255) -> float:
    """Calcula el SSIM (Structural Similarity Index Measure) entre dos imágenes.

    A diferencia de MSE/PSNR (que comparan píxel a píxel), SSIM modela
    la percepción humana comparando luminancia, contraste y estructura
    en ventanas locales. Rango: [-1, 1], donde 1.0 = imágenes idénticas.
    En la práctica, para fotografías naturales, SSIM suele estar en [0, 1].

    Parameters
    ----------
    original : np.ndarray
        Imagen de referencia (ground truth), uint8.
    comparison : np.ndarray
        Imagen a evaluar, uint8, mismo shape que `original`.
    data_range : int, default=255
        Rango dinámico de los valores de píxel (255 para uint8).

    Returns
    -------
    float
        Valor de SSIM en el rango [-1, 1] (1.0 = idénticas).
    """
    _validate_pair(original, comparison)
    channel_axis = -1 if original.ndim == 3 else None
    return float(
        skimage_ssim(original, comparison, data_range=data_range,
                     channel_axis=channel_axis)
    )


def evaluate_pair(original: np.ndarray, comparison: np.ndarray,
                   label: str = "comparison") -> dict:
    """Calcula MSE, PSNR y SSIM para un único par (original, comparison).

    Parameters
    ----------
    original : np.ndarray
        Imagen de referencia (ground truth), uint8.
    comparison : np.ndarray
        Imagen a evaluar, uint8.
    label : str, default="comparison"
        Etiqueta identificadora de la imagen evaluada (p. ej. "degradada",
        "restaurada_median", "restaurada_nlm+clahe").

    Returns
    -------
    dict
        Diccionario con las claves: "label", "MSE", "PSNR (dB)", "SSIM".
    """
    return {
        "label": label,
        "MSE": compute_mse(original, comparison),
        "PSNR (dB)": compute_psnr(original, comparison),
        "SSIM": compute_ssim(original, comparison),
    }


def build_comparison_table(original: np.ndarray,
                            candidates: dict[str, np.ndarray]) -> pd.DataFrame:
    """Construye una tabla comparativa de métricas para múltiples imágenes.

    Útil para comparar, en una sola tabla, la imagen degradada contra
    varias variantes restauradas (distintos filtros/pipelines), todas
    respecto al mismo ground truth.

    Parameters
    ----------
    original : np.ndarray
        Imagen de referencia (ground truth), uint8.
    candidates : dict[str, np.ndarray]
        Diccionario ``{etiqueta: imagen}`` con las versiones a evaluar,
        p. ej. ``{"degradada": img_deg, "median": img_med, "nlm+clahe": img_nc}``.

    Returns
    -------
    pd.DataFrame
        Tabla con columnas ``["label", "MSE", "PSNR (dB)", "SSIM"]``,
        ordenada de mejor a peor PSNR.

    Examples
    --------
    >>> table = build_comparison_table(
    ...     original=clean_img,
    ...     candidates={
    ...         "degradada": degraded_img,
    ...         "restaurada_median": restored_median,
    ...         "restaurada_nlm_clahe": restored_pipeline,
    ...     },
    ... )
    >>> print(table)
    """
    rows = [evaluate_pair(original, img, label=label)
            for label, img in candidates.items()]
    df = pd.DataFrame(rows)
    return df.sort_values(by="PSNR (dB)", ascending=False).reset_index(drop=True)