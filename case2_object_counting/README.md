# Caso 2 — Conteo Automático de Objetos

## Problema

Contar automáticamente objetos individuales en una imagen (partículas, colonias, monedas, etc.), incluyendo el caso más exigente en el que varios objetos se tocan o se superponen entre sí — escenario en el que los métodos de conteo más simples (contornos, componentes conexos) subestiman sistemáticamente el conteo real.

## Dataset

Se genera una imagen sintética de partículas circulares (`preprocessing.generate_synthetic_particles`) con solapamientos intencionales y un conteo real (ground truth) exactamente conocido, necesario para evaluar cuantitativamente la precisión de cada método sin depender de anotaciones manuales. Adicionalmente, el pipeline se valida sobre un dataset real: `skimage.data.coins()` (24 monedas fotografiadas).

## Pipeline

1. Preprocesamiento (`src/preprocessing.py`): generación de datos sintéticos con ground truth, conversión a escala de grises, denoising (gaussiano/mediana/bilateral) y realce de contraste (CLAHE).
2. Binarización y morfología (`src/segmentation.py`): umbral de Otsu (global) y umbral adaptativo (local); apertura, cierre, filtro por área mínima y eliminación de objetos que tocan el borde de la imagen; segmentación por Watershed para separar objetos superpuestos.
3. Detección de contornos/componentes (`src/detection.py`): contornos externos (`cv2.findContours`) y componentes conexos (`cv2.connectedComponentsWithStats`), con utilidades de visualización.
4. Conteo (`src/counting.py`): conteo por los tres métodos (contornos, componentes conexos, Watershed) y construcción de una tabla comparativa con error absoluto y relativo contra el ground truth.
5. Evaluación de precisión: comparación directa contra el conteo real conocido (imagen sintética) y validación cualitativa sobre el dataset real de monedas.

Detalle metodológico completo (justificación de cada técnica) en `docs/context_summary.md`, sección Caso 2.

## Resultados

Notebook completo y ejecutado: [`notebooks/case2_demo.ipynb`](./notebooks/case2_demo.ipynb).

Sobre la imagen sintética (40 partículas reales, con solapamientos intencionales):

| Método | Conteo | Ground truth | Error absoluto | Error relativo |
|---|---|---|---|---|
| **Watershed** | **35** | 40 | **5** | **12.5%** |
| Contornos externos | 28 | 40 | 12 | 30.0% |
| Componentes conexos | 28 | 40 | 12 | 30.0% |

Watershed separa correctamente partículas que se tocan, mientras que contornos y componentes conexos las cuentan como un único objeto, lo que explica la brecha de precisión. Tabla completa: [`results/metrics/case2_counting_comparison.csv`](./results/metrics/case2_counting_comparison.csv).

Sobre el dataset real `skimage.data.coins()` (24 monedas), aplicando CLAHE + Otsu + eliminación de componentes en el borde antes de Watershed, se detectaron correctamente **23 de 24 monedas completamente visibles** (la restante queda parcialmente cortada por el encuadre de la fotografía y se excluye del conteo a propósito). Figuras: [`results/figures/`](./results/figures/).

## Limitaciones

- El umbral adaptativo, si bien es más robusto ante iluminación no uniforme, resultó más sensible a la textura/ruido en este dataset y no se usó como método principal.
- Watershed no separa perfectamente partículas con solapamientos muy severos (un único máximo de distancia dominante puede seguir fusionando dos objetos muy próximos).
- Sobre datos reales (monedas), fue necesario un paso adicional de eliminación de objetos en el borde (`clear_border_objects`) para evitar que el gradiente de iluminación del fondo se fusionara con la máscara binaria — una limitación específica de un umbral global (Otsu) ante iluminación no perfectamente uniforme.
