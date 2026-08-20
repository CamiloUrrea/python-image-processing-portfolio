# Caso 3 — Reconocimiento de Placas Vehiculares en Tiempo Real

## Problema
[Describir el contexto: video en vivo, imágenes fijas, condiciones de iluminación, etc.]

## Dataset
[Fuente utilizada: CCPD / OpenALPR / AOLP / videos propios — ubicación esperada en `data/raw/`]

## Pipeline
1. Adquisición de video (`src/pipeline.py`)
2. Detección de la región de la placa (`src/detection.py`)
3. Preprocesamiento de la región detectada
4. OCR (`src/ocr.py`) — EasyOCR / Tesseract
5. Visualización del texto reconocido
6. Evaluación de desempeño (tasa de acierto, FPS)

## Resultados
Ver `results/figures/` y `results/metrics/`.

## Limitaciones
[Completar]
