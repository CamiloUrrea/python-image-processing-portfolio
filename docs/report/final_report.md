# Informe Final — Image Processing Portfolio

**Curso:** Python for Research — Módulo 1: Procesamiento de Imágenes
**Repositorio:** https://github.com/CamiloUrrea/python-image-processing-portfolio
**Autores:** [Nombre completo] — [correo institucional] · [Nombre completo] — [correo institucional] *(completar)*
**Fecha:** [Completar fecha de entrega]

> ⚠️ **Nota académica:** la consigna original del curso solicita desarrollar **un (1)** caso de estudio. Este informe documenta los tres casos como ejercicio de portafolio; si la entrega formal exige un único caso, indicar aquí cuál corresponde a la entrega oficial: **[Completar]**.

---

## Resumen

Este informe documenta el desarrollo de un portafolio de tres pipelines de procesamiento digital de imágenes implementados en Python con OpenCV y scikit-image: (1) restauración y mejora de fotografías degradadas, evaluada cuantitativamente con PSNR/SSIM; (2) conteo automático de objetos mediante binarización, morfología matemática y el algoritmo Watershed, comparado contra métodos de conteo por contornos y componentes conexos; y (3) reconocimiento de placas vehiculares (ANPR/OCR), combinando detección de regiones de interés basada en gradientes de Sobel con un motor de reconocimiento óptico de caracteres (EasyOCR). Los tres pipelines se validaron sobre datos sintéticos con *ground truth* conocido y, en los Casos 2 y 3, también sobre datos reales o parcialmente reales. Los resultados muestran que el pipeline combinado de restauración (Non-Local Means + CLAHE + realce de nitidez) mejora el PSNR de 18.30 dB a 19.83 dB frente a la imagen degradada; que la segmentación por Watershed reduce el error de conteo de objetos superpuestos del 30% al 12.5% frente a métodos de conteo directo; y que el pipeline de reconocimiento de placas alcanza un 100% de acierto exacto de texto sobre un lote de prueba de 6 frames sintéticos.

---

## 1. Introducción

El procesamiento digital de imágenes es un área central de la visión por computador, con aplicaciones que van desde la restauración de material fotográfico dañado hasta sistemas automatizados de conteo y de reconocimiento óptico de caracteres. Este proyecto explora tres problemas representativos del módulo de procesamiento de imágenes del curso, cada uno abordado con técnicas clásicas (no basadas en modelos de detección entrenados de extremo a extremo), lo que permite explicar y justificar cada etapa del pipeline de forma transparente.

Los tres casos comparten una misma filosofía metodológica: (a) definición precisa del problema, (b) construcción o selección de un conjunto de datos que permita medir el desempeño cuantitativamente, (c) implementación de un pipeline modular documentado, y (d) evaluación objetiva contra una referencia conocida (*ground truth*). Esta estructura común se refleja en la organización idéntica de las tres carpetas del repositorio (`src/`, `notebooks/`, `results/`).

## 2. Objetivos

### 2.1 Objetivo general

Diseñar, implementar y evaluar cuantitativamente tres pipelines de procesamiento digital de imágenes que aborden problemas representativos de restauración, segmentación/conteo y reconocimiento óptico de caracteres, utilizando técnicas clásicas de visión por computador.

### 2.2 Objetivos específicos

- Simular degradaciones fotográficas controladas y reproducibles, e implementar técnicas de restauración y realce que las reviertan, midiendo la mejora obtenida con métricas de fidelidad de imagen (MSE, PSNR, SSIM).
- Implementar y comparar métodos de binarización (Otsu, adaptativo), limpieza morfológica, y segmentación por Watershed para el conteo automático de objetos, cuantificando la ventaja de Watershed frente a métodos que no separan objetos superpuestos.
- Diseñar un pipeline de localización de placas vehiculares basado en gradientes morfológicos (sin modelos de detección entrenados) e integrarlo con un motor OCR mediante una interfaz modular intercambiable, evaluando la exactitud de la lectura de texto resultante.
- Documentar de forma reproducible cada pipeline en notebooks ejecutados de extremo a extremo, con 0 errores, y en un documento de contexto técnico complementario.

## 3. Metodología

### 3.1 Enfoque general

Los tres casos se implementaron como módulos Python independientes y autocontenidos (`src/`), cada uno con responsabilidades separadas por etapa del pipeline (preprocesamiento, técnica principal, evaluación/ensamblado). Cada caso incluye un notebook Jupyter de demostración que documenta visualmente el flujo completo, y una carpeta `results/` con las figuras y tablas de métricas generadas. Debido a la ausencia de datasets anotados de acceso inmediato en el entorno de desarrollo, los Casos 2 y 3 recurren a **datos sintéticos generados programáticamente con *ground truth* exacto conocido** (posiciones, conteos o textos reales), lo que permite una evaluación cuantitativa objetiva sin depender de anotación manual. El Caso 1 utiliza una imagen de referencia estándar de `scikit-image` (`data.astronaut()`) sobre la cual se simulan degradaciones controladas.

### 3.2 Caso 1 — Restauración y mejora de fotografías degradadas

Se simulan degradaciones reproducibles (semilla fija) sobre una imagen de referencia: ruido gaussiano, ruido sal y pimienta, desenfoque gaussiano y de movimiento, reducción de contraste, reducción de resolución y artefactos de compresión JPEG, individualmente y en pipelines combinados. Sobre la imagen degradada se aplican y comparan técnicas de restauración: filtrado gaussiano, de mediana, bilateral y Non-Local Means (Buades et al., 2005) para *denoising*; deconvolución de Wiener para revertir desenfoque; ecualización de histograma y CLAHE (Zuiderveld, 1994) para realce de contraste; y *unsharp masking* para realce de nitidez. La calidad de cada variante restaurada se mide contra la imagen original mediante MSE, PSNR y SSIM (Wang et al., 2004).

### 3.3 Caso 2 — Conteo automático de objetos

Se genera una imagen sintética de partículas circulares con solapamientos intencionales entre algunas de ellas, y un conteo real (*ground truth*) exacto. La imagen se binariza con el método de Otsu (Otsu, 1979) y, alternativamente, con umbral adaptativo local, y se limpia con operaciones morfológicas de apertura, cierre y filtrado por área mínima. Se comparan tres métodos de conteo: (a) contornos externos, (b) componentes conexos, y (c) segmentación por Watershed (Vincent & Soille, 1991) sobre la transformada de distancia, que separa explícitamente objetos adyacentes o superpuestos. El pipeline se valida adicionalmente sobre un dataset real (`skimage.data.coins()`), incorporando CLAHE y eliminación de componentes en el borde de la imagen para manejar iluminación no uniforme.

### 3.4 Caso 3 — Reconocimiento de placas vehiculares (ANPR/OCR)

La localización de la región de la placa se implementa con una técnica clásica de ANPR basada en gradientes: conversión a escala de grises, filtrado bilateral (preserva bordes), gradiente de Sobel en el eje horizontal (resalta las transiciones verticales características de los caracteres alfanuméricos), binarización de Otsu, cierre morfológico con un kernel rectangular ancho (funde los caracteres individuales en un bloque continuo) y filtrado de contornos candidatos por relación de aspecto y área. La región recortada se preprocesa (escala de grises, reescalado, filtrado de mediana) y se pasa a un motor OCR — EasyOCR, un modelo de deep learning para detección y reconocimiento de texto en escena — encapsulado detrás de una interfaz abstracta (`OCRReader`) que permite sustituir el motor sin modificar el resto del pipeline. El texto reconocido se normaliza (mayúsculas, filtrado de caracteres no alfanuméricos) y se selecciona la lectura de mayor confianza. Se generan frames sintéticos de un vehículo con placa (texto y bounding box de *ground truth* conocidos) para evaluar tanto la precisión de localización de la ROI como la exactitud del texto reconocido, sobre un lote que simula un flujo de video.

## 4. Resultados

### 4.1 Caso 1 — Restauración y mejora de fotografías degradadas

Sobre una degradación combinada (desenfoque gaussiano + ruido gaussiano + reducción de contraste), se evaluaron seis técnicas de restauración individuales y un pipeline combinado, contra la imagen degradada sin restaurar:

| Variante | MSE | PSNR (dB) | SSIM |
|---|---|---|---|
| **Pipeline combinado (NLM + CLAHE + Unsharp)** | **675.96** | **19.83** | **0.625** |
| Filtro de mediana | 848.28 | 18.85 | 0.595 |
| Deconvolución de Wiener | 851.73 | 18.83 | 0.512 |
| Filtro gaussiano | 885.55 | 18.66 | 0.601 |
| Filtro bilateral | 891.48 | 18.63 | 0.602 |
| Non-Local Means (solo) | 894.28 | 18.62 | 0.583 |
| Degradada (sin restaurar) | 962.12 | 18.30 | 0.384 |

*Fuente: `case1_restoration/results/metrics/case1_metrics_comparison.csv`.*

El pipeline combinado obtuvo el mejor resultado en las tres métricas simultáneamente, con una mejora de +1.53 dB en PSNR y +0.241 en SSIM respecto a la imagen degradada sin restaurar.

### 4.2 Caso 2 — Conteo automático de objetos

Sobre la imagen sintética (40 partículas reales, con solapamientos intencionales):

| Método | Conteo | Ground truth | Error absoluto | Error relativo |
|---|---|---|---|---|
| **Watershed** | **35** | 40 | **5** | **12.5%** |
| Contornos externos | 28 | 40 | 12 | 30.0% |
| Componentes conexos | 28 | 40 | 12 | 30.0% |

*Fuente: `case2_object_counting/results/metrics/case2_counting_comparison.csv`.*

Watershed redujo el error relativo de conteo en 17.5 puntos porcentuales frente a los métodos que no separan objetos superpuestos. Sobre el dataset real `skimage.data.coins()` (24 monedas), el pipeline con CLAHE + Otsu + eliminación de componentes en el borde detectó correctamente 23 de 24 monedas completamente visibles (la restante queda parcialmente cortada por el encuadre de la fotografía, excluida intencionalmente).

### 4.3 Caso 3 — Reconocimiento de placas vehiculares (ANPR/OCR)

Sobre un lote de 6 frames sintéticos con placas distintas (simulación de flujo de video):

| Ground truth | Texto detectado | Confianza | Acierto |
|---|---|---|---|
| ABC1234 | ABC1234 | 0.992 | Sí |
| XYZ4821 | XYZ4821 | 0.816 | Sí |
| JKL9087 | JKL9087 | 0.998 | Sí |
| QRS5566 | QRS5566 | 0.991 | Sí |
| TUV3321 | TUV3321 | 0.988 | Sí |
| MNO7712 | MNO7712 | 0.266 | Sí |

*Fuente: `case3_license_plate/results/metrics/case3_ocr_accuracy.csv`.*

**Tasa de acierto exacto: 100% (6/6)**, con confianzas del motor OCR entre 0.27 y 1.00. La detección de la ROI de la placa sobre el frame de prueba obtuvo un bounding box de `(247, 292, 147, 42)` frente a un *ground truth* de `(250, 291, 140, 43)`, con una desviación de pocos píxeles.

## 5. Discusión

Los tres casos ilustran un patrón común: combinar varias técnicas clásicas en un pipeline supera consistentemente a aplicar una técnica aislada. En el Caso 1, ninguna técnica individual de restauración superó al pipeline combinado (denoising + realce de contraste + nitidez) en las tres métricas simultáneamente. En el Caso 2, la diferencia entre un pipeline "ingenuo" (binarización + contornos) y uno que incorpora Watershed fue la diferencia entre subestimar sistemáticamente el conteo real y aproximarse a él. En el Caso 3, la combinación de una técnica de detección clásica (gradientes de Sobel) con un motor OCR moderno basado en deep learning demostró ser efectiva sin necesitar un detector de objetos entrenado específicamente para placas.

Un hallazgo metodológico transversal fue la sensibilidad de los umbrales globales (Otsu) a la iluminación no uniforme: tanto en el Caso 2 (dataset de monedas) como en el diseño del Caso 3 se identificó que un único umbral global puede fusionar artefactos de iluminación con el fondo, lo que se corrigió con CLAHE (contraste local) y con la eliminación de componentes conectados al borde de la imagen — una decisión de preprocesamiento no trivial que quedó documentada en detalle en `docs/context_summary.md`.

También se identificó que la preparación de la imagen de entrada debe adaptarse al algoritmo consumidor: mientras que la binarización dura (Otsu) es beneficiosa para el conteo de objetos por contornos, resultó contraproducente como entrada directa al motor OCR basado en deep learning del Caso 3, que rinde mejor con imágenes en escala de grises de gradiente continuo.

## 6. Conclusiones

- El pipeline de restauración combinado (Non-Local Means + CLAHE + Unsharp Masking) fue superior a cualquier técnica individual evaluada, confirmando que denoising, realce de contraste y realce de nitidez atacan degradaciones complementarias y no sustituibles entre sí.
- La segmentación por Watershed es indispensable quando el conteo de objetos debe manejar solapamientos: redujo el error relativo de conteo del 30% al 12.5% en el escenario sintético evaluado, y generalizó razonablemente bien a un dataset real de monedas.
- Un pipeline de detección clásico basado en gradientes morfológicos (sin redes de detección entrenadas) fue suficiente para localizar una región de placa vehicular con precisión de pocos píxeles, y su combinación con un motor OCR modular alcanzó 100% de exactitud de texto en el lote de prueba sintético evaluado.
- El diseño modular (interfaces abstractas, funciones puras, pipelines configurables por lista de pasos) usado en los tres casos facilitó tanto la experimentación con distintas combinaciones de técnicas como la extensión futura (p. ej. sustituir el motor OCR, o añadir nuevas degradaciones/técnicas de restauración) sin reescribir el resto del sistema.
- Como trabajo futuro, la validación de los tres pipelines contra datasets reales más diversos (fotografías degradadas reales, imágenes de conteo con anotación manual, fotografías reales de vehículos y placas) es el paso pendiente más relevante para confirmar que las conclusiones obtenidas sobre datos sintéticos generalizan a condiciones de uso reales.

## 7. Video explicativo

- **URL de YouTube:** `[Completar — pegar aquí el enlace del video una vez publicado]`
- **Código QR:**

```
[Completar — insertar aquí la imagen del código QR que enlaza al video, p. ej. docs/media/qr_video.png]
```

## 8. Referencias (formato APA)

Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.

Buades, A., Coll, B., & Morel, J. M. (2005). A non-local algorithm for image denoising. *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)*, *2*, 60–65. https://doi.org/10.1109/CVPR.2005.38

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, *585*, 357–362. https://doi.org/10.1038/s41586-020-2649-2

JaidedAI. (2023). *EasyOCR* [Software]. GitHub. https://github.com/JaidedAI/EasyOCR

Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, *9*(1), 62–66. https://doi.org/10.1109/TSMC.1979.4310076

van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014). scikit-image: Image processing in Python. *PeerJ*, *2*, e453. https://doi.org/10.7717/peerj.453

Vincent, L., & Soille, P. (1991). Watersheds in digital spaces: An efficient algorithm based on immersion simulations. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *13*(6), 583–598. https://doi.org/10.1109/34.87344

Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, *13*(4), 600–612. https://doi.org/10.1109/TIP.2003.819861

Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. S. Heckbert (Ed.), *Graphics Gems IV* (pp. 474–485). Academic Press.
