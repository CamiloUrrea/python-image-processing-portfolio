"""
segmentation.py — Caso 2: Conteo Automático de Objetos

Funciones de:
    1. Binarización        -> umbral de Otsu (global) y umbral adaptativo (local)
    2. Operaciones morfológicas -> limpieza de la máscara binaria
    3. Watershed            -> separación de objetos que se tocan/superponen

Convención: las máscaras binarias son arrays uint8 con valores {0, 255},
donde 255 representa "objeto (primer plano)" y 0 representa "fondo".

Autor: Portafolio Python for Research — Módulo 2
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as skimage_watershed
from skimage.segmentation import clear_border as skimage_clear_border


# ----------------------------------------------------------------------
# 1. Binarización
# ----------------------------------------------------------------------

def otsu_threshold(gray_image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Binariza una imagen usando el umbral global de Otsu.

    El método de Otsu calcula automáticamente el umbral que minimiza la
    varianza intra-clase (objeto vs. fondo), asumiendo una distribución
    de intensidad bimodal. Es rápido y funciona bien cuando la
    iluminación es razonablemente uniforme sobre toda la imagen.

    Parameters
    ----------
    gray_image : np.ndarray
        Imagen en escala de grises (uint8).
    invert : bool, default=False
        Si es True, invierte la máscara (útil cuando el objeto de
        interés es más oscuro que el fondo).

    Returns
    -------
    np.ndarray
        Máscara binaria (uint8, valores {0, 255}).
    """
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray_image, 0, 255, flag + cv2.THRESH_OTSU)
    return mask


def adaptive_threshold(gray_image: np.ndarray, method: str = "gaussian",
                        block_size: int = 25, C: float = 5.0,
                        invert: bool = False) -> np.ndarray:
    """Binariza una imagen usando un umbral adaptativo (local).

    A diferencia de Otsu (un único umbral global), el umbral adaptativo
    calcula un umbral distinto para cada píxel en función de sus vecinos
    (media o media ponderada gaussiana de un bloque local menos una
    constante C). Es más robusto ante iluminación no uniforme, a costa
    de ser más sensible a ruido de alta frecuencia.

    Parameters
    ----------
    gray_image : np.ndarray
        Imagen en escala de grises (uint8).
    method : str, default="gaussian"
        ``"gaussian"`` (media ponderada gaussiana del vecindario) o
        ``"mean"`` (media simple del vecindario).
    block_size : int, default=25
        Tamaño del vecindario local usado para calcular el umbral
        (debe ser impar y > 1).
    C : float, default=5.0
        Constante restada de la media/media ponderada calculada.
    invert : bool, default=False
        Si es True, invierte la máscara (objeto más oscuro que el fondo).

    Returns
    -------
    np.ndarray
        Máscara binaria (uint8, valores {0, 255}).
    """
    if block_size % 2 == 0:
        raise ValueError("block_size debe ser un número impar.")

    adaptive_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
                        else cv2.ADAPTIVE_THRESH_MEAN_C)
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    return cv2.adaptiveThreshold(
        gray_image, 255, adaptive_method, flag, block_size, C
    )


# ----------------------------------------------------------------------
# 2. Operaciones morfológicas
# ----------------------------------------------------------------------

def _structuring_element(kernel_size: int, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    """Construye un elemento estructurante para las operaciones morfológicas."""
    return cv2.getStructuringElement(shape, (kernel_size, kernel_size))


def morphological_opening(binary: np.ndarray, kernel_size: int = 3,
                           iterations: int = 1) -> np.ndarray:
    """Aplica apertura morfológica (erosión seguida de dilatación).

    Elimina pequeños objetos/ruido aislado en el fondo (falsos positivos)
    sin alterar significativamente el tamaño de los objetos grandes.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).
    kernel_size : int, default=3
        Tamaño del elemento estructurante (elíptico).
    iterations : int, default=1
        Número de veces que se aplica la operación.

    Returns
    -------
    np.ndarray
        Máscara binaria filtrada (uint8).
    """
    kernel = _structuring_element(kernel_size)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)


def morphological_closing(binary: np.ndarray, kernel_size: int = 3,
                           iterations: int = 1) -> np.ndarray:
    """Aplica cierre morfológico (dilatación seguida de erosión).

    Rellena pequeños huecos dentro de los objetos y une regiones
    cercanas que deberían pertenecer al mismo objeto, sin alterar
    significativamente su tamaño.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).
    kernel_size : int, default=3
        Tamaño del elemento estructurante (elíptico).
    iterations : int, default=1
        Número de veces que se aplica la operación.

    Returns
    -------
    np.ndarray
        Máscara binaria filtrada (uint8).
    """
    kernel = _structuring_element(kernel_size)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def remove_small_objects(binary: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Elimina componentes conexos con área menor a ``min_area``.

    Complementa a la apertura morfológica: descarta explícitamente
    manchas de ruido residual por debajo de un área mínima, sin importar
    su forma (a diferencia de la apertura, que depende del kernel).

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).
    min_area : int, default=20
        Área mínima (en píxeles) que debe tener un componente para
        conservarse.

    Returns
    -------
    np.ndarray
        Máscara binaria filtrada (uint8).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label_id in range(1, num_labels):  # 0 = fondo
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def clear_border_objects(binary: np.ndarray) -> np.ndarray:
    """Elimina de la máscara los objetos que tocan el borde de la imagen.

    Práctica estándar en conteo de objetos sobre datos reales: un objeto
    cortado por el borde del encuadre no está completo (su forma/área
    real es desconocida) y, además, artefactos de iluminación no
    uniforme (viñeteo, gradientes de brillo) suelen fusionarse con el
    fondo precisamente en los bordes de la imagen, generando blobs
    espurios. Excluir los componentes conectados al borde resuelve
    ambos problemas de forma simple.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}).

    Returns
    -------
    np.ndarray
        Máscara binaria (uint8, {0, 255}) sin los componentes que
        tocaban el borde.
    """
    cleared_bool = skimage_clear_border(binary > 0)
    return (cleared_bool.astype(np.uint8)) * 255


def clean_binary_mask(binary: np.ndarray, kernel_size: int = 3,
                       open_iterations: int = 1, close_iterations: int = 1,
                       min_area: int | None = 20) -> np.ndarray:
    """Pipeline corto de limpieza morfológica: apertura + cierre + filtro por área.

    Orden recomendado: apertura (elimina ruido puntual) -> cierre
    (rellena huecos internos) -> filtro por área mínima (elimina
    componentes residuales que sobrevivieron a la apertura).

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria de entrada (uint8, {0, 255}).
    kernel_size : int, default=3
        Tamaño del elemento estructurante usado en apertura y cierre.
    open_iterations : int, default=1
        Iteraciones de la apertura.
    close_iterations : int, default=1
        Iteraciones del cierre.
    min_area : int, optional, default=20
        Área mínima para conservar un componente. Si es ``None``, se
        omite este paso.

    Returns
    -------
    np.ndarray
        Máscara binaria limpia (uint8).
    """
    result = morphological_opening(binary, kernel_size, open_iterations)
    result = morphological_closing(result, kernel_size, close_iterations)
    if min_area is not None:
        result = remove_small_objects(result, min_area)
    return result


# ----------------------------------------------------------------------
# 3. Watershed (separación de objetos que se tocan/superponen)
# ----------------------------------------------------------------------

def watershed_segmentation(binary: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """Separa objetos adyacentes/superpuestos en una máscara binaria vía Watershed.

    A diferencia de contornos o componentes conexos (que tratan un
    grupo de objetos que se tocan como un único blob), Watershed usa la
    transformada de distancia para encontrar los "centros" (máximos
    locales, más alejados del borde) de cada objeto y hace crecer
    regiones desde esos centros hasta que se encuentran, generando así
    fronteras entre objetos individuales dentro de un mismo blob.

    Algoritmo:
        1. Transformada de distancia sobre la máscara binaria.
        2. Detección de máximos locales de la distancia (= semillas,
           un máximo por objeto).
        3. Watershed sobre la distancia negada, usando las semillas
           como marcadores.

    Parameters
    ----------
    binary : np.ndarray
        Máscara binaria (uint8, {0, 255}) con los objetos en primer
        plano (255).
    min_distance : int, default=10
        Distancia mínima (en píxeles) entre dos máximos locales para
        considerarse semillas de objetos distintos. Controla la
        sensibilidad de la separación: valores bajos separan objetos
        muy cercanos pero pueden sobre-segmentar; valores altos son
        más conservadores.

    Returns
    -------
    np.ndarray
        Imagen etiquetada (int32) del mismo shape que ``binary``, donde
        cada objeto individual tiene una etiqueta entera única (>= 1) y
        el fondo tiene etiqueta 0.
    """
    binary_bool = binary > 0
    distance = ndi.distance_transform_edt(binary_bool)

    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_bool)
    seed_mask = np.zeros(distance.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(seed_mask)

    labels = skimage_watershed(-distance, markers, mask=binary_bool)
    return labels.astype(np.int32)
