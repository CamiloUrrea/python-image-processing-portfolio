"""
restoration.py — Caso 1: Restauración y Mejora de Fotografías Degradadas

Funciones de RESTAURACIÓN (denoising / deconvolución) y de REALCE
(contraste / nitidez) aplicables sobre imágenes degradadas.

Organización:
    1. Denoising / filtrado espacial   -> reduce ruido preservando estructura
    2. Deconvolución                    -> revierte desenfoque conocido
    3. Realce de contraste              -> mejora rango dinámico
    4. Realce de nitidez                -> resalta bordes/detalle
    5. Pipeline combinado               -> encadena varias técnicas

Autor: Portafolio Python for Research — Módulo 1
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import cv2
from skimage.restoration import wiener, unsupervised_wiener


# ----------------------------------------------------------------------
# 1. Denoising / filtrado espacial
# ----------------------------------------------------------------------

def gaussian_filter(image: np.ndarray, kernel_size: tuple[int, int] = (5, 5),
                     sigma: float = 0.0) -> np.ndarray:
    """Suaviza la imagen con un filtro Gaussiano (reduce ruido Gaussiano leve).

    Trade-off: reduce ruido pero también difumina bordes finos, ya que
    pondera todos los píxeles vecinos sin distinguir estructura.

    Parameters
    ----------
    image : np.ndarray
        Imagen degradada (uint8).
    kernel_size : tuple[int, int], default=(5, 5)
        Tamaño del kernel (valores impares).
    sigma : float, default=0.0
        Desviación estándar; 0 = calculada automáticamente por OpenCV.

    Returns
    -------
    np.ndarray
        Imagen filtrada (uint8).
    """
    return cv2.GaussianBlur(image, kernel_size, sigma)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Aplica un filtro de mediana (muy efectivo contra ruido sal y pimienta).

    A diferencia del filtro Gaussiano, el filtro de mediana no promedia
    sino que reemplaza cada píxel por la mediana de su vecindario, lo cual
    elimina valores atípicos (impulsos) sin difuminar tanto los bordes.

    Parameters
    ----------
    image : np.ndarray
        Imagen degradada (uint8).
    kernel_size : int, default=5
        Tamaño del kernel (debe ser impar y > 1).

    Returns
    -------
    np.ndarray
        Imagen filtrada (uint8).
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size debe ser un número impar.")
    return cv2.medianBlur(image, kernel_size)


def bilateral_filter(image: np.ndarray, d: int = 9,
                      sigma_color: float = 75.0,
                      sigma_space: float = 75.0) -> np.ndarray:
    """Aplica un filtro bilateral (suaviza preservando bordes).

    Combina un kernel espacial (como el Gaussiano) con un kernel de
    intensidad: solo promedia píxeles vecinos que además son similares
    en valor, por lo que preserva bordes mejor que Gaussiano/mediana.
    Es más costoso computacionalmente.

    Parameters
    ----------
    image : np.ndarray
        Imagen degradada (uint8).
    d : int, default=9
        Diámetro del vecindario de píxeles considerado.
    sigma_color : float, default=75.0
        Filtro de rango: a mayor valor, colores más distintos se mezclan.
    sigma_space : float, default=75.0
        Filtro espacial: a mayor valor, píxeles más lejanos se influencian.

    Returns
    -------
    np.ndarray
        Imagen filtrada (uint8).
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def non_local_means_denoise(image: np.ndarray, h: float = 10.0,
                             template_window_size: int = 7,
                             search_window_size: int = 21) -> np.ndarray:
    """Aplica denoising Non-Local Means (NLM), estado del arte en filtrado clásico.

    En lugar de comparar solo píxeles vecinos, NLM busca en toda la imagen
    parches similares al parche actual y los promedia. Es más lento pero
    preserva texturas y bordes notablemente mejor que los filtros locales.

    Parameters
    ----------
    image : np.ndarray
        Imagen degradada (uint8), color (BGR) o escala de grises.
    h : float, default=10.0
        Fuerza del filtrado. A mayor h, más ruido se elimina pero también
        más detalle se pierde.
    template_window_size : int, default=7
        Tamaño del parche usado para comparar similitud (impar).
    search_window_size : int, default=21
        Tamaño de la ventana de búsqueda de parches similares (impar).

    Returns
    -------
    np.ndarray
        Imagen filtrada (uint8).
    """
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(
            image, None, h, h, template_window_size, search_window_size
        )
    return cv2.fastNlMeansDenoising(
        image, None, h, template_window_size, search_window_size
    )


# ----------------------------------------------------------------------
# 2. Deconvolución (reversión de desenfoque)
# ----------------------------------------------------------------------

def _gaussian_psf(size: int = 15, sigma: float = 3.0) -> np.ndarray:
    """Construye una PSF (point spread function) Gaussiana normalizada."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return psf / psf.sum()


def wiener_deconvolution(image: np.ndarray, psf: np.ndarray | None = None,
                          balance: float = 0.1) -> np.ndarray:
    """Revierte un desenfoque conocido (o estimado) mediante filtro de Wiener.

    A diferencia de un filtro de denoising, la deconvolución intenta
    invertir explícitamente el proceso de degradación (convolución con
    una PSF de desenfoque), por lo que puede recuperar detalle perdido
    por blur — a costa de amplificar ruido si `balance` es muy bajo.

    Nota: opera sobre imágenes en escala de grises (o canal por canal
    si se le pasa una imagen a color).

    Parameters
    ----------
    image : np.ndarray
        Imagen degradada (uint8), escala de grises o color.
    psf : np.ndarray, optional
        Point Spread Function que modela el desenfoque. Si es None, se
        usa una PSF Gaussiana genérica como aproximación.
    balance : float, default=0.1
        Parámetro de regularización del filtro de Wiener (equivalente a
        1/SNR). Valores más altos = más suavizado / menos ruido amplificado.

    Returns
    -------
    np.ndarray
        Imagen restaurada (uint8).
    """
    if psf is None:
        psf = _gaussian_psf(size=15, sigma=3.0)

    def _deconv_channel(channel: np.ndarray) -> np.ndarray:
        img_float = channel.astype(np.float64) / 255.0
        restored = wiener(img_float, psf, balance=balance, clip=True)
        return np.clip(restored * 255.0, 0, 255)

    if image.ndim == 3:
        channels = cv2.split(image)
        restored_channels = [_deconv_channel(ch).astype(np.uint8) for ch in channels]
        return cv2.merge(restored_channels)

    return _deconv_channel(image).astype(np.uint8)


# ----------------------------------------------------------------------
# 3. Realce de contraste
# ----------------------------------------------------------------------

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Aplica ecualización de histograma global para mejorar el contraste.

    Redistribuye la intensidad de los píxeles para que el histograma
    resultante sea aproximadamente uniforme. Efectivo en imágenes con
    bajo contraste global, pero puede sobre-amplificar ruido y generar
    resultados poco naturales en imágenes con iluminación no uniforme.

    Parameters
    ----------
    image : np.ndarray
        Imagen degradada (uint8), escala de grises o color (BGR).

    Returns
    -------
    np.ndarray
        Imagen con contraste realzado (uint8).
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    # Para color: ecualizar solo el canal de luminancia (evita distorsión de color)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0,
                       tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    A diferencia de la ecualización global, CLAHE opera por regiones
    (tiles) y limita la amplificación de contraste (clip_limit), evitando
    la sobre-amplificación de ruido típica de la ecualización clásica.
    Generalmente produce resultados más naturales.

    Parameters
    ----------
    image : np.ndarray
        Imagen degradada (uint8), escala de grises o color (BGR).
    clip_limit : float, default=2.0
        Límite de contraste por tile (a mayor valor, más contraste/ruido).
    tile_grid_size : tuple[int, int], default=(8, 8)
        Número de tiles (filas, columnas) en que se divide la imagen.

    Returns
    -------
    np.ndarray
        Imagen con contraste realzado (uint8).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------
# 4. Realce de nitidez
# ----------------------------------------------------------------------

def unsharp_mask(image: np.ndarray, sigma: float = 1.0,
                  strength: float = 1.5) -> np.ndarray:
    """Aplica realce de nitidez mediante Unsharp Masking.

    Técnica clásica: se genera una versión desenfocada de la imagen,
    se calcula la "máscara" (diferencia entre original y desenfocada,
    que resalta bordes/detalle de alta frecuencia), y se suma de vuelta
    a la imagen original amplificada por `strength`.

    Parameters
    ----------
    image : np.ndarray
        Imagen de entrada (uint8).
    sigma : float, default=1.0
        Desviación estándar del desenfoque Gaussiano usado para generar
        la máscara. Controla el "radio" del realce.
    strength : float, default=1.5
        Cantidad de realce aplicado (1.0 = sin cambio, >1.0 = más nítido).

    Returns
    -------
    np.ndarray
        Imagen con nitidez realzada (uint8).
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, strength, blurred, 1.0 - strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# 5. Pipeline combinado
# ----------------------------------------------------------------------

RESTORATION_REGISTRY = {
    "gaussian_filter": gaussian_filter,
    "median_filter": median_filter,
    "bilateral_filter": bilateral_filter,
    "non_local_means": non_local_means_denoise,
    "wiener_deconvolution": wiener_deconvolution,
    "histogram_equalization": histogram_equalization,
    "clahe": clahe_enhancement,
    "unsharp_mask": unsharp_mask,
}


def apply_restoration_pipeline(image: np.ndarray,
                                steps: list[tuple[str, dict]]) -> np.ndarray:
    """Aplica una secuencia de técnicas de restauración/realce en cadena.

    Recomendación general de orden: primero denoising/deconvolución,
    luego realce de contraste, y por último nitidez — aplicar nitidez
    antes de eliminar ruido tiende a amplificar el propio ruido.

    Parameters
    ----------
    image : np.ndarray
        Imagen degradada de entrada (uint8).
    steps : list[tuple[str, dict]]
        Lista de pasos ``(nombre_tecnica, kwargs)`` en el orden a aplicar.
        ``nombre_tecnica`` debe existir en ``RESTORATION_REGISTRY``.

    Returns
    -------
    np.ndarray
        Imagen restaurada resultante (uint8).

    Examples
    --------
    >>> steps = [
    ...     ("non_local_means", {"h": 10}),
    ...     ("clahe", {"clip_limit": 2.0}),
    ...     ("unsharp_mask", {"strength": 1.3}),
    ... ]
    >>> restored = apply_restoration_pipeline(degraded_image, steps)
    """
    result = image.copy()
    for name, kwargs in steps:
        if name not in RESTORATION_REGISTRY:
            raise ValueError(
                f"Técnica de restauración desconocida: '{name}'. "
                f"Opciones válidas: {list(RESTORATION_REGISTRY.keys())}"
            )
        result = RESTORATION_REGISTRY[name](result, **kwargs)
    return result


# ----------------------------------------------------------------------
# 6. Interfaz de línea de comandos (CLI)
# ----------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        prog="restoration.py",
        description=(
            "Restaura una imagen degradada aplicando una o más técnicas de "
            "denoising/deconvolución y realce en cadena."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Ruta de la imagen degradada de entrada.")
    parser.add_argument("output", type=str, help="Ruta donde guardar la imagen restaurada.")
    parser.add_argument(
        "--technique", "-t",
        action="append",
        dest="techniques",
        choices=sorted(RESTORATION_REGISTRY.keys()),
        help=(
            "Técnica a aplicar (repetible para encadenar varias, en el orden "
            "indicado). Si se omite, se usa el pipeline por defecto: "
            "non_local_means -> clahe -> unsharp_mask."
        ),
    )
    parser.add_argument("--nlm-h", type=float, default=10.0,
                         help="Fuerza del filtro Non-Local Means.")
    parser.add_argument("--median-kernel", type=int, default=5,
                         help="Tamaño de kernel del filtro de mediana (impar).")
    parser.add_argument("--gaussian-kernel", type=int, default=5,
                         help="Tamaño de kernel del filtro Gaussiano (impar).")
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                         help="Clip limit de CLAHE.")
    parser.add_argument("--unsharp-strength", type=float, default=1.5,
                         help="Fuerza del realce de nitidez (unsharp mask).")
    return parser


def _kwargs_for_technique(technique: str, args: argparse.Namespace) -> dict:
    """Traduce los argumentos de la CLI a los kwargs de cada técnica."""
    mapping = {
        "non_local_means": {"h": args.nlm_h},
        "median_filter": {"kernel_size": args.median_kernel},
        "gaussian_filter": {"kernel_size": (args.gaussian_kernel, args.gaussian_kernel)},
        "clahe": {"clip_limit": args.clahe_clip},
        "unsharp_mask": {"strength": args.unsharp_strength},
    }
    return mapping.get(technique, {})


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la CLI.

    Uso:
        python restoration.py <input> <output> [--technique NOMBRE ...] [opciones]

    Ejemplo:
        python restoration.py degradada.jpg restaurada.jpg \\
            --technique non_local_means --technique clahe --technique unsharp_mask
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        parser.error(f"No se pudo leer la imagen de entrada: '{args.input}'")

    techniques = args.techniques or ["non_local_means", "clahe", "unsharp_mask"]
    steps = [(name, _kwargs_for_technique(name, args)) for name in techniques]

    print(f"Aplicando pipeline de restauración: {' -> '.join(techniques)}")
    restored = apply_restoration_pipeline(image, steps)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not cv2.imwrite(args.output, restored):
        raise RuntimeError(f"No se pudo guardar la imagen restaurada en: '{args.output}'")

    print(f"Imagen restaurada guardada en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())