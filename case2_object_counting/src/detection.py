"""
detection.py — Caso 2: Conteo Automático de Objetos

Funciones de detección de objetos individuales sobre una máscara binaria
(o una imagen etiquetada por Watershed), mediante dos enfoques clásicos:

    - Contornos (`cv2.findContours`): encuentra las fronteras de cada
      blob. Simple y rápido, pero trata objetos que se tocan como uno solo.
    - Componentes conexos (`cv2.connectedComponentsWithStats`): etiqueta
      cada blob 4/8-conectado y entrega estadísticas (área, bounding
      box, centroide) directamente.

Ambos métodos comparten la misma limitación: no separan objetos
superpuestos/tocándose (para eso se usa Watershed, ver `segmentation.py`).

También incluye utilidades de visualización (dibujar contornos,
componentes o regiones de Watershed sobre la imagen original).

Autor: Portafolio Python for Research — Módulo 2
"""

from __future__ import annotations

import numpy as np
import cv2


# ----------------------------------------------------------------------
# Contornos
# ----------------------------------------------------------------------

def find_contours(binary: np.ndarray, min_area: float = 0.0,
                   max_area: float | None = None) -> list[np.ndarray]:
    """Encuentra los contornos externos de los objetos en una máscara binaria.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).
    min_area : float, default=0.0
        Área mínima (píxeles) para conservar un contorno. Descarta
        ruido residual muy pequeño.
    max_area : float, optional
        Área máxima (píxeles) para conservar un contorno. Útil para
        descartar blobs anómalamente grandes (p. ej. varios objetos
        fusionados que Watershed no pudo separar).

    Returns
    -------
    list[np.ndarray]
        Lista de contornos (cada uno un array de puntos), filtrados por
        área y ordenados de mayor a menor área.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        filtered.append(cnt)

    return sorted(filtered, key=cv2.contourArea, reverse=True)


def draw_contours(image: np.ndarray, contours: list[np.ndarray],
                   color: tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2, label_index: bool = True) -> np.ndarray:
    """Dibuja contornos sobre una copia de la imagen (para visualización).

    Parameters
    ----------
    image : np.ndarray
        Imagen base (uint8), color (BGR) o escala de grises (se
        convierte internamente a BGR para poder dibujar en color).
    contours : list[np.ndarray]
        Contornos a dibujar (p. ej. salida de :func:`find_contours`).
    color : tuple[int, int, int], default=(0, 255, 0)
        Color BGR de los contornos.
    thickness : int, default=2
        Grosor de línea.
    label_index : bool, default=True
        Si es True, numera cada contorno con su índice (1-based) sobre
        su centroide.

    Returns
    -------
    np.ndarray
        Copia de la imagen (BGR, uint8) con los contornos dibujados.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    cv2.drawContours(vis, contours, -1, color, thickness)

    if label_index:
        for i, cnt in enumerate(contours, start=1):
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.putText(vis, str(i), (cx - 6, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return vis


# ----------------------------------------------------------------------
# Componentes conexos
# ----------------------------------------------------------------------

def connected_components(binary: np.ndarray, connectivity: int = 8,
                          min_area: float = 0.0) -> dict:
    """Etiqueta componentes conexos en una máscara binaria y filtra por área.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).
    connectivity : int, default=8
        Conectividad usada para agrupar píxeles vecinos (4 u 8).
    min_area : float, default=0.0
        Área mínima (píxeles) para conservar un componente.

    Returns
    -------
    dict
        Diccionario con:
            - ``"num_objects"``: número de componentes tras el filtro.
            - ``"labels"``: imagen etiquetada (int32), reindexada tras
              el filtrado (0 = fondo).
            - ``"stats"``: array (N, 5) con [x, y, w, h, área] por objeto.
            - ``"centroids"``: array (N, 2) con el centroide (x, y) de
              cada objeto.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )

    keep_ids = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    new_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(keep_ids, start=1):
        new_labels[labels == old_id] = new_id

    return {
        "num_objects": len(keep_ids),
        "labels": new_labels,
        "stats": stats[keep_ids],
        "centroids": centroids[keep_ids],
    }


def draw_labeled_regions(image: np.ndarray, labels: np.ndarray,
                          boundary_color: tuple[int, int, int] = (0, 0, 255),
                          label_index: bool = True) -> np.ndarray:
    """Dibuja fronteras y numera las regiones de una imagen etiquetada.

    Compatible tanto con la salida de :func:`connected_components`
    (``labels``) como con la de ``segmentation.watershed_segmentation``.

    Parameters
    ----------
    image : np.ndarray
        Imagen base (uint8), color (BGR) o escala de grises.
    labels : np.ndarray
        Imagen etiquetada (enteros), 0 = fondo, >=1 = un objeto por
        etiqueta distinta.
    boundary_color : tuple[int, int, int], default=(0, 0, 255)
        Color BGR de las fronteras entre regiones.
    label_index : bool, default=True
        Si es True, numera cada región sobre su centroide.

    Returns
    -------
    np.ndarray
        Copia de la imagen (BGR, uint8) con fronteras y etiquetas.
    """
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    # Fronteras: dilatar cada máscara de región y restar la máscara original
    # para dibujar solo el borde de cada componente etiquetado.
    for region_id in range(1, labels.max() + 1):
        region_mask = (labels == region_id).astype(np.uint8) * 255
        if not region_mask.any():
            continue
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, boundary_color, 1)

        if label_index:
            ys, xs = np.where(labels == region_id)
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(vis, str(region_id), (cx - 6, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, boundary_color, 1, cv2.LINE_AA)

    return vis
