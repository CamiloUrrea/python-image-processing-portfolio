# Caso 3 — Reconocimiento de Placas Vehiculares (ANPR/OCR)

## Problema

Localizar automáticamente la región de una placa vehicular dentro de una imagen o frame de video, y leer su texto alfanumérico mediante OCR, superponiendo el resultado (bounding box + texto + confianza) sobre la imagen original. Se aborda con visión por computador clásica (sin un detector de objetos entrenado) para la localización de la placa, y un motor OCR basado en deep learning para la lectura del texto.

## Dataset

Se generan frames sintéticos de un vehículo con una placa (`pipeline.generate_synthetic_vehicle_frame`): un rectángulo de color simulando la carrocería, una placa blanca con texto renderizado y ruido gaussiano, con ground truth exacto (texto real y bounding box real). No se dispuso de un dataset real de fotografías de vehículos/placas en el entorno de desarrollo; datasets sugeridos para una validación futura: CCPD, OpenALPR Benchmark, AOLP, o videos autograbados, ubicados en `data/raw/`.

## Pipeline

1. Detección de la región de la placa (`src/detection.py`): escala de grises → filtro bilateral (preserva bordes) → gradiente de Sobel en el eje X (resalta las transiciones verticales de los caracteres) → binarización de Otsu → cierre morfológico con kernel rectangular ancho (fusiona caracteres en un bloque) → refinamiento por erosión/dilatación → filtrado de contornos candidatos por relación de aspecto y área.
2. Preprocesamiento de la región detectada (`src/ocr.py`, `preprocess_plate_roi`): recorte, escala de grises, reescalado (upscale), reducción de ruido (mediana); se calcula también la binarización de Otsu con fines comparativos, aunque no se usa como entrada al OCR (ver `docs/context_summary.md`, sección 6.2, para la justificación).
3. OCR (`src/ocr.py`): interfaz abstracta `OCRReader` con una implementación `EasyOCRReader`, y post-procesamiento de limpieza alfanumérica del texto reconocido.
4. Ensamblado del pipeline completo (`src/pipeline.py`): `process_frame` para una imagen, `process_video_frames` para una secuencia de frames, y `draw_plate_result` para la visualización del texto reconocido sobre el bounding box.
5. Evaluación de desempeño: tasa de acierto exacto de texto contra el ground truth conocido, sobre un lote de frames sintéticos.

Detalle metodológico completo (justificación de cada etapa, decisiones de preprocesamiento para OCR, arquitectura modular) en `docs/context_summary.md`, sección Caso 3.

## Resultados

Notebook completo y ejecutado: [`notebooks/case3_demo.ipynb`](./notebooks/case3_demo.ipynb).

Sobre un lote de 6 frames sintéticos con placas distintas (simulación de flujo de video):

| Ground truth | Texto detectado | Confianza | Acierto |
|---|---|---|---|
| ABC1234 | ABC1234 | 0.992 | ✅ |
| XYZ4821 | XYZ4821 | 0.816 | ✅ |
| JKL9087 | JKL9087 | 0.998 | ✅ |
| QRS5566 | QRS5566 | 0.991 | ✅ |
| TUV3321 | TUV3321 | 0.988 | ✅ |
| MNO7712 | MNO7712 | 0.266 | ✅ |

**Tasa de acierto exacto: 100% (6/6)**, con confianzas del motor OCR entre 0.27 y 1.00. Tabla completa: [`results/metrics/case3_ocr_accuracy.csv`](./results/metrics/case3_ocr_accuracy.csv). Figuras (etapas de detección, preprocesamiento y resultado final): [`results/figures/`](./results/figures/).

## Limitaciones

- Validado exclusivamente con datos sintéticos (texto renderizado, fondos de color sólido); no se probó contra fotografías reales ni datasets públicos de ANPR, que introducen perspectiva, iluminación no uniforme, reflejos y fuentes tipográficas reales.
- El detector de ROI asume una placa rectangular sin rotación significativa; una inclinación pronunciada saca la relación de aspecto medida fuera del rango válido.
- `EasyOCRReader` se evaluó únicamente en CPU; no se midió desempeño ni tiempos de inferencia en GPU.
- No hay tracking entre frames de un mismo video: cada frame se procesa de forma independiente, sin deduplicar lecturas de una misma placa a lo largo del tiempo.
- No se midió FPS ni desempeño en tiempo real sobre un flujo de video continuo (cámara o archivo de video), solo sobre imágenes/frames estáticos procesados de forma individual.
