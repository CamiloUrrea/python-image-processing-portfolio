# Resumen de contexto — image-processing-portfolio

> Documento de contexto para retomar el trabajo (p. ej. con otro asistente
> como Gemini). No es el informe académico final (ver `docs/report/`).
> Generado: 2026-08-20. Última actualización: 2026-08-20 (Caso 3).

## 0. Regla de estilo de código vigente desde el Caso 3

A partir del Caso 3 (inclusive), todo el código fuente (`.py` y celdas
de código de notebooks) se escribe **sin comentarios explicativos en
línea** — solo se permiten docstrings esenciales en firmas de
funciones/clases cuando aportan información no evidente por el nombre
o la anotación de tipos. Toda la justificación técnica, el flujo
lógico y las decisiones metodológicas se documentan exclusivamente en
este archivo (`docs/context_summary.md`), no en el código. Esta regla
**no es retroactiva**: los módulos de los Casos 1 y 2 conservan sus
docstrings extensos en estilo NumPy/Google tal como se escribieron
originalmente.

## 1. Qué es este proyecto

Repositorio académico (curso *Python for Research*, módulo de
procesamiento de imágenes) que agrupa **3 casos de estudio** de visión
por computador, cada uno con su propio pipeline en `src/`, notebook de
demostración en `notebooks/` y resultados en `results/`:

| Caso | Carpeta | Estado |
|---|---|---|
| 1. Restauración y mejora de fotografías degradadas | `case1_restoration/` | ✅ Completo |
| 2. Conteo automático de objetos | `case2_object_counting/` | ✅ Completo |
| 3. Reconocimiento de placas (ANPR/OCR) | `case3_license_plate/` | ✅ Completo |

Repositorio remoto: `https://github.com/CamiloUrrea/python-image-processing-portfolio.git`
(rama `main`).

Nota de la propia guía académica (ver `README.md` raíz): la consigna
original del curso pide seleccionar **un (1)** caso; este repo agrupa
los tres como ejercicio de portafolio.

## 2. Historial de commits relevante

```
3b2d909  feat: initial repository architecture and configuration   (scaffold vacío)
05cd1a3  feat(case1): implementar pipeline de restauración de imágenes + CLI + notebook demo
bf9b59a  feat(case2): implementar pipeline de conteo automático de objetos + notebook demo
<HEAD>    feat(case3): implementar pipeline de reconocimiento de placas (OCR), notebook demo y contexto tecnico
```

## 3. Entorno de trabajo

- Entorno virtual local en `.venv/` (ignorado por git), creado con
  Python 3.14. Paquetes clave instalados: `numpy`, `scipy`, `pandas`,
  `opencv-python`, `scikit-image`, `matplotlib`, `jupyter`,
  `nbconvert`, `nbclient`, `ipykernel`.
- `requirements.txt` en la raíz declara el set completo de
  dependencias del proyecto.
- Para el Caso 3 se instaló adicionalmente `easyocr` (1.7.2), que trajo
  consigo `torch` (2.13.0, build CPU) como dependencia. En el primer
  uso, `easyocr.Reader(["en"])` descarga automáticamente los modelos de
  detección y reconocimiento de texto (pesos pre-entrenados) desde los
  servidores de JaidedAI y los cachea localmente (requiere conexión a
  internet la primera vez; ejecuciones posteriores usan el caché local
  y no vuelven a descargar). `pytesseract` quedó declarado en
  `requirements.txt` pero **no se usó**: requiere el binario del motor
  Tesseract instalado a nivel de sistema operativo, no disponible en
  este entorno; la abstracción modular de `ocr.py` permite añadirlo
  como backend alternativo en el futuro sin tocar `pipeline.py`.
- Los tres notebooks se ejecutaron de punta a punta con
  `jupyter nbconvert --execute --inplace` y se verificó
  programáticamente que **0 celdas** contienen `output_type == "error"`.

## 4. Caso 1 — Restauración y mejora de fotografías degradadas

**Objetivo:** simular degradaciones controladas sobre una imagen limpia
(ground truth), aplicar técnicas de restauración/realce, y medir
cuantitativamente la mejora con métricas objetivas (PSNR/SSIM).

### Módulos (`case1_restoration/src/`)

| Módulo | Función dentro del pipeline |
|---|---|
| `preprocessing.py` | Simula degradaciones controladas y reproducibles sobre una imagen limpia: ruido Gaussiano, ruido sal-y-pimienta, desenfoque Gaussiano, desenfoque de movimiento, reducción de contraste, reducción de resolución, artefactos JPEG. Incluye `apply_degradation_pipeline` para encadenar varias degradaciones (escenario realista). Este módulo genera los pares *(original, degradada)* necesarios para evaluar objetivamente la restauración. |
| `restoration.py` | Implementa las técnicas de restauración/realce: filtros de denoising (Gaussiano, mediana, bilateral, Non-Local Means), deconvolución de Wiener (revierte desenfoque conocido), realce de contraste (ecualización de histograma, CLAHE) y de nitidez (unsharp masking). Incluye `apply_restoration_pipeline` para encadenar técnicas, y un `RESTORATION_REGISTRY` central. **Expone una CLI (`argparse`)** al final del archivo: `python restoration.py <input> <output> [--technique NOMBRE ...] [opciones]`, con pipeline por defecto `non_local_means → clahe → unsharp_mask` si no se especifican técnicas. |
| `evaluation.py` | Métricas de calidad *full-reference* (requieren la imagen original como ground truth): MSE, PSNR (dB) y SSIM, usando `skimage.metrics`. `build_comparison_table` arma una tabla `pandas` comparando múltiples variantes (degradada + cada técnica de restauración) contra el original, ordenada de mejor a peor PSNR. |

### Notebook: `case1_restoration/notebooks/case1_demo.ipynb`

Flujo: carga `skimage.data.astronaut()` como ground truth → aplica 4
degradaciones individuales + 1 pipeline combinado (blur+ruido+bajo
contraste) → prueba 5 técnicas de restauración individuales + 1 pipeline
combinado (NLM + CLAHE + Unsharp) → construye tabla comparativa de
métricas → guarda figuras y CSV.

**Resultado principal:** el pipeline combinado (NLM+CLAHE+Unsharp) obtuvo
el mejor PSNR/SSIM de todas las variantes evaluadas (PSNR ≈ 19.83 dB,
SSIM ≈ 0.625, vs. 18.30 dB / 0.384 de la imagen degradada sin restaurar).

### Resultados guardados

- `results/figures/01..05_*.png` (5 figuras)
- `results/metrics/case1_metrics_comparison.csv`

## 5. Caso 2 — Conteo automático de objetos

**Objetivo:** contar objetos individuales en una imagen mediante
binarización + limpieza morfológica + detección, comparando métodos
ingenuos (contornos, componentes conexos) contra Watershed (que sí
separa objetos que se tocan/superponen), evaluando contra un ground
truth conocido.

### Módulos (`case2_object_counting/src/`)

| Módulo | Función dentro del pipeline |
|---|---|
| `preprocessing.py` | `generate_synthetic_particles(...)`: genera una imagen sintética de círculos ("partículas") con solapamientos intencionales y devuelve el **conteo real (ground truth)** junto con centros/radios — necesario porque no siempre hay datasets anotados disponibles. Además: `to_grayscale`, `denoise` (gaussian/median/bilateral), `enhance_contrast` (CLAHE), y `prepare_for_segmentation` (pipeline corto: grises → denoising → CLAHE opcional). |
| `segmentation.py` | **Binarización:** `otsu_threshold` (umbral global automático) y `adaptive_threshold` (umbral local, robusto a iluminación no uniforme pero más sensible a ruido/textura). **Morfología:** `morphological_opening`, `morphological_closing`, `remove_small_objects`, `clean_binary_mask` (pipeline: apertura → cierre → filtro de área mínima). `clear_border_objects`: descarta componentes que tocan el borde de la imagen (objetos parciales / artefactos de iluminación en el borde — práctica estándar en datos reales). **Watershed:** `watershed_segmentation` — usa transformada de distancia + detección de máximos locales (semillas) + `skimage.segmentation.watershed` para separar objetos individuales dentro de un mismo blob. |
| `detection.py` | `find_contours` (contornos externos vía `cv2.findContours`, filtrables por área) y `connected_components` (etiquetado vía `cv2.connectedComponentsWithStats`, con stats de área/bbox/centroide). Ambos comparten la limitación de no separar objetos que se tocan. Incluye utilidades de visualización: `draw_contours` y `draw_labeled_regions` (esta última también sirve para pintar la salida de Watershed). |
| `counting.py` | Envuelve los métodos anteriores en `count_by_contours`, `count_by_connected_components`, `count_by_watershed`, cada uno devolviendo `{"count": int, ...}`. `build_counting_report(results, ground_truth)` arma una tabla `pandas` comparando el conteo de cada método contra el ground truth, con error absoluto y relativo (%), ordenada por menor error. |

### Notebook: `case2_object_counting/notebooks/case2_demo.ipynb`

Flujo: genera imagen sintética (40 partículas, solapamientos
intencionales, ground truth conocido) → preprocesa → compara
binarización Otsu vs. adaptativa → limpieza morfológica → detecta por
contornos y por componentes conexos → separa objetos superpuestos con
Watershed → tabla comparativa final de conteo vs. ground truth → aplica
el mismo pipeline a un dataset real (`skimage.data.coins`, 24 monedas)
para validar generalización.

**Resultado principal (imagen sintética, ground truth = 40):**

| Método | Conteo | Error absoluto | Error relativo |
|---|---|---|---|
| Watershed | 35 | 5 | 12.5% |
| Contornos | 28 | 12 | 30.0% |
| Componentes conexos | 28 | 12 | 30.0% |

Watershed se acerca notablemente más al ground truth al separar
partículas solapadas que los otros dos métodos fusionan en un único
objeto.

**Dataset real (`skimage.data.coins`, 24 monedas):** un Otsu "ingenuo"
fusionaba el gradiente de iluminación del fondo con las monedas del
borde superior (blob espurio conectado al borde). Se corrigió aplicando
CLAHE antes de Otsu y descartando componentes que tocan el borde
(`clear_border_objects`) — resultado final: 23 monedas completamente
visibles detectadas correctamente; la moneda restante queda cortada por
el encuadre y se excluye a propósito (práctica estándar).

### Resultados guardados

- `results/figures/01..08_*.png` (8 figuras)
- `results/metrics/case2_counting_comparison.csv`

## 6. Caso 3 — Reconocimiento de placas vehiculares (ANPR/OCR)

**Objetivo:** localizar automáticamente la región de una placa
vehicular dentro de una imagen/frame (sin usar un detector basado en
deep learning), preprocesarla, leer su texto con un motor OCR y
superponer visualmente el resultado (bounding box + texto + confianza)
sobre el frame original. El diseño contempla tanto el procesamiento de
una imagen aislada como el de una secuencia de frames (simulación de
flujo de video).

### 6.1 Justificación metodológica de la arquitectura de detección

Se implementó la técnica clásica de ANPR (*Automatic Number Plate
Recognition*) basada en gradientes morfológicos, anterior a los
detectores de objetos basados en redes neuronales, por ser
representativa del enfoque de "visión por computador clásica" que
articula todo el portafolio (Casos 1, 2 y 3 usan exclusivamente
técnicas clásicas de OpenCV/scikit-image, sin modelos de detección
entrenados). El razonamiento detrás de cada etapa:

1. **Escala de grises** (`detection.to_grayscale`): la geometría de la
   placa (bordes, caracteres) no depende del color.
2. **Filtro bilateral** (`detection.denoise_edge_preserving`): reduce
   ruido de textura de la carrocería del vehículo sin difuminar los
   bordes verticales de la placa y sus caracteres, que son la señal
   clave del siguiente paso. Se prefirió sobre un filtro Gaussiano
   exactamente por esta razón (igual que en `case1_restoration`, un
   filtro bilateral preserva bordes mejor que uno puramente espacial).
3. **Gradiente de Sobel en X** (`detection.sobel_gradient_magnitude`):
   las placas contienen una alta densidad de transiciones verticales
   de intensidad (bordes de caracteres alfanuméricos y del marco de la
   placa), muy superior a la de la carrocería o el fondo circundantes.
   Un gradiente horizontal (Sobel X) resalta selectivamente esa
   región, actuando como un detector de "textura de texto".
4. **Binarización de Otsu sobre el gradiente**
   (`detection.binarize_gradient`): convierte el mapa de gradiente
   continuo en una máscara binaria, aplicando el mismo criterio de
   umbral automático global usado en el Caso 2 (aquí sobre la magnitud
   del gradiente, no sobre la intensidad original).
5. **Cierre morfológico con kernel rectangular ancho (17×3)**
   (`detection.close_plate_regions`): los caracteres individuales
   generan segmentos de gradiente discontinuos; un kernel
   deliberadamente más ancho que alto fusiona esos segmentos
   horizontalmente contiguos en un único blob rectangular que cubre
   toda la placa, sin fusionarse verticalmente con estructuras por
   encima o debajo (p. ej. parachoques, sombras).
6. **Refinamiento erosión + dilatación**
   (`detection.refine_candidate_mask`): limpia irregularidades del
   contorno del blob resultante del cierre, antes de extraer contornos.
7. **Filtrado de contornos por relación de aspecto y área**
   (`detection.find_plate_candidates`): de todos los blobs candidatos,
   se conservan solo los que tienen una relación de aspecto ancho/alto
   entre 2.0 y 6.5 (rango típico de placas rectangulares) y un área
   entre un mínimo absoluto (300 px, descarta ruido) y un máximo
   relativo al área total de la imagen (25%, descarta blobs
   gigantes como el propio vehículo). Los candidatos se ordenan por
   área descendente — el más grande dentro del rango válido es la
   mejor estimación de la placa real.

`detection.extract_roi` recorta la región del frame original
correspondiente al mejor candidato, con un margen de padding de 4 px
para no cortar el borde de la placa.

### 6.2 Pipeline de preprocesamiento de la ROI para OCR

Implementado en `ocr.preprocess_plate_roi`, con una decisión
metodológica deliberada: **el motor OCR (EasyOCR) no recibe la máscara
binaria como entrada**, sino la versión en escala de grises,
reescalada (upscale ×2 por defecto) y suavizada con un filtro de
mediana (`ocr.denoise_roi`). Esto se debe a que EasyOCR usa modelos de
deep learning (CRAFT para detección de texto + CRNN para
reconocimiento) entrenados sobre imágenes de texto en escenas
naturales con gradientes continuos y anti-aliasing; una binarización
dura (Otsu) elimina esa información de gradiente y en la práctica
reduce la confianza/precisión del reconocimiento en modelos de este
tipo, a diferencia de los pipelines de OCR clásico basados en patrones
de píxeles binarios (p. ej. Tesseract clásico), donde la binarización
sí suele ayudar. Aun así, `preprocess_plate_roi` calcula y expone la
versión binarizada (`ocr.binarize_roi`, vía Otsu) en su diccionario de
retorno — se muestra en el notebook con fines metodológicos y
comparativos, y queda disponible para experimentar con un backend OCR
alternativo (p. ej. Tesseract) que sí se beneficie de ella.

### 6.3 Abstracción modular del motor OCR (`ocr.py`)

Se definió una interfaz abstracta `OCRReader` (`abc.ABC`) con un único
método `read(image) -> list[dict]`, y una implementación concreta
`EasyOCRReader` que envuelve `easyocr.Reader`. `pipeline.py` depende
únicamente de la interfaz `OCRReader`, nunca de EasyOCR directamente
(salvo en `build_default_ocr_reader`, el único punto de construcción
concreto) — esto permite sustituir el motor OCR (p. ej. por un backend
basado en Tesseract/pytesseract, o un servicio en la nube) sin
modificar `pipeline.py` ni `detection.py`, implementando simplemente
una nueva subclase de `OCRReader`.

**Post-procesamiento de texto** (`ocr.clean_plate_text`): normaliza a
mayúsculas y elimina cualquier carácter que no sea alfanumérico
(regex `[^A-Z0-9]`), ya que las placas reales no contienen espacios,
guiones ni símbolos, pero EasyOCR puede insertarlos por artefactos de
segmentación de texto. `ocr.select_best_reading` filtra por confianza
mínima y selecciona la lectura de mayor confianza cuando el motor OCR
devuelve múltiples fragmentos de texto detectados en la ROI (esto
ocurre con frecuencia si el detector de texto interno de EasyOCR
fragmenta la placa en más de un bloque).

### 6.4 Ensamblado del pipeline (`pipeline.py`)

- `build_default_ocr_reader`: punto único de construcción de un
  `EasyOCRReader` (idioma inglés por defecto, CPU).
- `process_frame(frame, reader)`: ejecuta
  `detection.locate_plate_candidates` → toma el mejor candidato →
  `detection.extract_roi` → `ocr.preprocess_plate_roi` →
  `ocr.read_plate_text`. Devuelve un diccionario con `bbox`, `text`,
  `confidence`, la ROI cruda, la ROI preprocesada y la lista completa
  de candidatos (para depuración/visualización). Si no se detecta
  ningún candidato válido, devuelve `success: False` sin lanzar
  excepción — se considera un resultado válido del dominio (placa no
  encontrada), no un error de programación.
- `process_video_frames(frames, reader)`: aplica `process_frame` a
  cada elemento de una lista de frames, para representar el
  procesamiento de un flujo de video. **No implementa tracking entre
  frames** (cada frame se procesa de forma completamente
  independiente) ni captura de cámara real — en este entorno sin
  acceso a hardware de video, la "secuencia de video" se simula con
  varios frames sintéticos independientes generados por
  `generate_synthetic_vehicle_frame`.
- `draw_plate_result(frame, result)`: dibuja el bounding box verde y
  una etiqueta con el texto reconocido + confianza sobre una copia del
  frame original.
- `generate_synthetic_vehicle_frame(...)`: utilidad de generación de
  datos sintéticos (mismo patrón metodológico que
  `case2_object_counting/src/preprocessing.py`), necesaria porque no
  se dispone de un dataset real de fotografías de vehículos/placas en
  este entorno. Dibuja un rectángulo "carrocería" de color aleatorio y
  una placa blanca con borde negro y texto (`cv2.putText`) en su
  parte inferior central, añade ruido Gaussiano, y devuelve tanto el
  frame como el ground truth exacto (`plate_text` y `bbox` real de la
  placa) — permite medir cuantitativamente tanto la precisión de
  localización de la ROI como la exactitud del texto reconocido.

### 6.5 Notebook: `case3_license_plate/notebooks/case3_demo.ipynb`

Flujo (10 celdas de código, ejecutadas con 0 errores): configuración
del entorno (incluye la carga del `EasyOCRReader`, que descarga los
modelos en la primera ejecución) → generación de un frame sintético →
visualización de las 6 etapas internas de detección de ROI (grises →
bilateral → Sobel → Otsu → cierre → refinada) → visualización de los
candidatos detectados con su bounding box → visualización de las 4
etapas de preprocesamiento de la ROI para OCR → lectura OCR cruda y
texto limpio → resultado final anotado sobre el frame → validación por
lote sobre 6 frames sintéticos con placas distintas (simulación de
video) → tabla de exactitud guardada en CSV.

### 6.6 Resultados obtenidos

- **Detección de ROI:** sobre el frame de prueba (`XYZ4821`), el único
  candidato detectado tuvo bbox `(247, 292, 147, 42)` frente al ground
  truth `(250, 291, 140, 43)` — desviación de pocos píxeles, atribuible
  al padding y a la naturaleza aproximada del cierre morfológico.
- **Lectura OCR y exactitud sobre 6 frames sintéticos** (tabla completa
  en `results/metrics/case3_ocr_accuracy.csv`):

  | ground_truth | detected_text | confidence | match |
  |---|---|---|---|
  | ABC1234 | ABC1234 | 0.9919 | True |
  | XYZ4821 | XYZ4821 | 0.8158 | True |
  | JKL9087 | JKL9087 | 0.9981 | True |
  | QRS5566 | QRS5566 | 0.9905 | True |
  | TUV3321 | TUV3321 | 0.9882 | True |
  | MNO7712 | MNO7712 | 0.2656 | True |

  **Tasa de acierto exacto: 100% (6/6)**, con confianzas del motor OCR
  entre 0.27 y 1.00. El caso de menor confianza (`MNO7712`, 0.27)
  acertó igualmente el texto completo — la confianza reportada por
  EasyOCR refleja la certeza interna del modelo de reconocimiento de
  caracteres (sensible a la combinación aleatoria de color de fondo y
  ruido Gaussiano generada para ese frame), no necesariamente la
  corrección del resultado final; en un sistema de producción, este
  campo se usaría como criterio para descartar o re-procesar lecturas
  de baja confianza, incluso si en esta muestra particular el
  resultado fue correcto.

### 6.7 Limitaciones conocidas

- Todo el pipeline se validó **exclusivamente con datos sintéticos**
  (texto renderizado con `cv2.putText`, fondos y carrocerías de color
  sólido con ruido Gaussiano). No se probó contra fotografías reales
  de vehículos/placas ni contra datasets públicos de ANPR (p. ej.
  OpenALPR, CCPD), que introducirían variables no simuladas aquí:
  perspectiva/inclinación de la placa, iluminación no uniforme y
  reflejos, oclusiones parciales, desenfoque de movimiento, y fuentes
  tipográficas reales de placas vehiculares.
- El detector de ROI asume una placa **rectangular sin rotación
  significativa**; una placa inclinada más de unos pocos grados
  reduciría la relación de aspecto medida por `cv2.boundingRect` fuera
  del rango válido, o el gradiente de Sobel en X dejaría de alinearse
  con los bordes de los caracteres.
- `EasyOCRReader` se ejecutó únicamente en **CPU** (`gpu=False`); no se
  validó desempeño ni tiempos de inferencia en GPU.
- No se implementó de-duplicación/tracking de una misma placa a través
  de múltiples frames consecutivos de un video real (cada frame se
  trata de forma independiente).

## 7. Pendiente / no iniciado

- `shared/utils.py` y `shared/visualization.py`: siguen siendo
  placeholders, no utilizados por ningún caso (los tres casos son
  autocontenidos, sin dependencias cruzadas entre carpetas `src/`).
- Los `README.md` internos de cada caso (`case1_restoration/README.md`,
  `case2_object_counting/README.md`, `case3_license_plate/README.md`)
  mantienen su texto de plantilla original (`[Describir...]`,
  `[Completar]`) — no se actualizaron con los resultados reales
  obtenidos.
- El informe académico final formal (`docs/report/`) sigue vacío.
- Validación del pipeline del Caso 3 contra datos reales (fotografías
  de vehículos/placas), pendiente por falta de dataset disponible en
  este entorno (ver limitaciones, sección 6.7).
