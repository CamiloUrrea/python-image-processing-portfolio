"""
counting.py — Caso 2: Conteo Automático de Objetos

Funciones de alto nivel que envuelven `detection.py` y
`segmentation.py` para producir un conteo final de objetos, junto con
utilidades para comparar la precisión de distintos métodos de conteo
contra un valor de referencia (ground truth), cuando este es conocido
(p. ej. en imágenes sintéticas generadas por `preprocessing.py`).

Métodos de conteo comparados:
    - Por contornos externos.
    - Por componentes conexos.
    - Por Watershed (separa objetos que se tocan/superponen).

Autor: Portafolio Python for Research — Módulo 2
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import detection as det
import segmentation as seg


def count_by_contours(binary: np.ndarray, min_area: float = 20.0) -> dict:
    """Cuenta objetos a partir de contornos externos.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).
    min_area : float, default=20.0
        Área mínima (píxeles) para conservar un contorno.

    Returns
    -------
    dict
        ``{"count": int, "contours": list[np.ndarray]}``.
    """
    contours = det.find_contours(binary, min_area=min_area)
    return {"count": len(contours), "contours": contours}


def count_by_connected_components(binary: np.ndarray, connectivity: int = 8,
                                   min_area: float = 20.0) -> dict:
    """Cuenta objetos a partir de componentes conexos.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).
    connectivity : int, default=8
        Conectividad usada para agrupar píxeles vecinos (4 u 8).
    min_area : float, default=20.0
        Área mínima (píxeles) para conservar un componente.

    Returns
    -------
    dict
        Resultado de :func:`detection.connected_components`, con al
        menos la clave ``"count"`` (alias de ``"num_objects"``).
    """
    result = det.connected_components(binary, connectivity=connectivity, min_area=min_area)
    result["count"] = result["num_objects"]
    return result


def count_by_watershed(binary: np.ndarray, min_distance: int = 10) -> dict:
    """Cuenta objetos separando blobs adyacentes/superpuestos vía Watershed.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).
    min_distance : int, default=10
        Distancia mínima entre semillas (ver
        :func:`segmentation.watershed_segmentation`).

    Returns
    -------
    dict
        ``{"count": int, "labels": np.ndarray}``.
    """
    labels = seg.watershed_segmentation(binary, min_distance=min_distance)
    count = int(labels.max())  # 0 = fondo; las etiquetas son 1..N
    return {"count": count, "labels": labels}


def build_counting_report(results: dict[str, int], ground_truth: int | None = None) -> pd.DataFrame:
    """Construye una tabla comparativa de conteo por método.

    Parameters
    ----------
    results : dict[str, int]
        Diccionario ``{nombre_metodo: conteo_obtenido}``.
    ground_truth : int, optional
        Conteo real de referencia. Si se provee, se añaden columnas de
        error absoluto y error relativo (%) respecto al ground truth.

    Returns
    -------
    pd.DataFrame
        Tabla con columnas ``["method", "count"]`` y, si aplica,
        ``["ground_truth", "abs_error", "rel_error_%"]``, ordenada por
        menor error absoluto (o por conteo si no hay ground truth).

    Examples
    --------
    >>> report = build_counting_report(
    ...     {"contornos": 34, "componentes_conexos": 34, "watershed": 41},
    ...     ground_truth=40,
    ... )
    """
    rows = [{"method": name, "count": count} for name, count in results.items()]
    df = pd.DataFrame(rows)

    if ground_truth is not None:
        df["ground_truth"] = ground_truth
        df["abs_error"] = (df["count"] - ground_truth).abs()
        df["rel_error_%"] = (df["abs_error"] / ground_truth * 100).round(2)
        return df.sort_values(by="abs_error").reset_index(drop=True)

    return df.sort_values(by="count", ascending=False).reset_index(drop=True)
