# Caso 1 — Restauración y Mejora de Fotografías Degradadas

## Problema

Restaurar fotografías afectadas por degradaciones comunes de adquisición y transmisión: ruido gaussiano, ruido impulsivo (sal y pimienta), desenfoque (gaussiano y de movimiento), pérdida de contraste, baja resolución y artefactos de compresión JPEG. El objetivo es recuperar la mayor fidelidad posible respecto a la imagen original, medida cuantitativamente.

## Dataset

Se utiliza `skimage.data.astronaut()` como imagen de referencia (ground truth) limpia. Al partir de una imagen sin degradar, se simulan degradaciones controladas y reproducibles (semillas fijas) sobre ella, lo que permite construir pares *(original, degradada)* y evaluar objetivamente la restauración sin depender de un dataset externo. El pipeline (`preprocessing.py`) también acepta cualquier imagen propia colocada en `data/raw/`.

## Pipeline

1. Simulación de degradaciones controladas (`src/preprocessing.py`): ruido gaussiano, sal y pimienta, desenfoque gaussiano/de movimiento, reducción de contraste, reducción de resolución, artefactos JPEG, y un pipeline combinado de varias degradaciones encadenadas.
2. Restauración (`src/restoration.py`): filtros de denoising (gaussiano, mediana, bilateral, Non-Local Means), deconvolución de Wiener, realce de contraste (ecualización de histograma, CLAHE) y de nitidez (unsharp masking), encadenables en un pipeline. Expone también una CLI (`python restoration.py <input> <output> [--technique ...]`).
3. Evaluación con métricas objetivas (`src/evaluation.py`): MSE, PSNR y SSIM contra la imagen original, con una tabla comparativa ordenada por PSNR.

Detalle metodológico completo (justificación de cada técnica y de las decisiones de diseño) en `docs/context_summary.md`, sección Caso 1.

## Resultados

Notebook completo y ejecutado: [`notebooks/case1_demo.ipynb`](./notebooks/case1_demo.ipynb).

Sobre una degradación combinada (desenfoque + ruido gaussiano + bajo contraste), el pipeline de restauración combinado (Non-Local Means → CLAHE → Unsharp Masking) fue el que mejor desempeño obtuvo entre todas las técnicas evaluadas:

| Variante | MSE | PSNR (dB) | SSIM |
|---|---|---|---|
| **Pipeline combinado (NLM + CLAHE + Unsharp)** | **675.96** | **19.83** | **0.625** |
| Filtro de mediana | 848.28 | 18.85 | 0.595 |
| Deconvolución de Wiener | 851.73 | 18.83 | 0.512 |
| Filtro gaussiano | 885.55 | 18.66 | 0.601 |
| Filtro bilateral | 891.48 | 18.63 | 0.602 |
| Non-Local Means (solo) | 894.28 | 18.62 | 0.583 |
| Degradada (sin restaurar) | 962.12 | 18.30 | 0.384 |

Tabla completa: [`results/metrics/case1_metrics_comparison.csv`](./results/metrics/case1_metrics_comparison.csv). Figuras: [`results/figures/`](./results/figures/).

## Limitaciones

- Evaluado sobre una única imagen de referencia (`skimage.data.astronaut()`); no se validó sobre un dataset diverso de fotografías reales degradadas.
- La deconvolución de Wiener asume una PSF (función de dispersión de punto) gaussiana genérica cuando no se conoce la real, lo que limita su efectividad frente a desenfoques de movimiento u ópticos no gaussianos.
- Las métricas PSNR/SSIM son *full-reference* (requieren la imagen original); no aplican directamente a fotografías degradadas reales sin un ground truth conocido.
