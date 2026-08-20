# 🖼️ Image Processing Portfolio — Python for Research (Módulo 1)

Portafolio de procesamiento de imágenes desarrollado en Python, que implementa **tres casos de estudio** de visión por computador propuestos en la guía del proyecto del Módulo 1. Aunque la consigna original solicita seleccionar un único caso, este repositorio se construyó como un portafolio técnico integral que demuestra dominio de las distintas técnicas del módulo.

> ⚠️ **Nota académica:** para la entrega formal del curso, revisar los lineamientos del profesor — la consigna original exige seleccionar **un (1)** caso de estudio. Este repositorio agrupa los tres como ejercicio de portafolio; el informe (`docs/report/`) debe dejar claro cuál caso corresponde a la entrega oficial si así lo requiere la evaluación.

---

## 📋 Tabla de contenidos

- [Casos de estudio](#-casos-de-estudio)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Instalación](#-instalación)
- [Ejecución](#-ejecución)
- [Metodología general](#-metodología-general)
- [Resultados](#-resultados)
- [Dependencias principales](#-dependencias-principales)
- [Datasets sugeridos](#-datasets-sugeridos)
- [Autores](#-autores)
- [Referencias](#-referencias)

---

## 🎯 Casos de estudio

| # | Caso | Objetivo | Carpeta |
|---|------|----------|---------|
| 1 | **Restauración y mejora de fotografías degradadas** | Restaurar/mejorar imágenes degradadas (ruido, desenfoque, baja resolución, artefactos) | [`case1_restoration/`](./case1_restoration) |
| 2 | **Conteo automático de objetos** | Detectar y contar objetos en imágenes digitales mediante segmentación y análisis de contornos | [`case2_object_counting/`](./case2_object_counting) |
| 3 | **Reconocimiento de placas vehiculares en tiempo real** | Detectar placas en video/imagen y aplicar OCR para extraer el texto | [`case3_license_plate/`](./case3_license_plate) |

Cada caso cuenta con su propio `README.md` interno con detalles específicos de la metodología, resultados y limitaciones.

---

## 📁 Estructura del repositorio

```
image-processing-portfolio/
│
├── README.md                      # Este archivo
├── requirements.txt                # Dependencias del proyecto
├── .gitignore
├── LICENSE
│
├── docs/
│   ├── report/                     # Informe final (PDF) exigido por la guía
│   └── media/                      # Video explicativo (link/QR) y capturas
│
├── shared/                         # Código reutilizado entre los 3 casos
│   ├── utils.py                    # Funciones auxiliares (I/O, métricas comunes)
│   └── visualization.py            # Comparaciones antes/después, grillas, plots
│
├── case1_restoration/
│   ├── src/
│   │   ├── preprocessing.py        # Análisis y preprocesamiento de la degradación
│   │   ├── restoration.py          # Pipeline de restauración
│   │   └── evaluation.py           # Métricas (PSNR, SSIM, etc.)
│   ├── notebooks/
│   │   └── demo_restoration.ipynb  # Notebook de demostración end-to-end
│   ├── data/
│   │   ├── raw/                    # Imágenes degradadas de entrada (ignorado en git)
│   │   └── processed/              # Imágenes restauradas (ignorado en git)
│   └── results/
│       ├── figures/                # Comparaciones visuales
│       └── metrics/                # Resultados cuantitativos (CSV/JSON)
│
├── case2_object_counting/
│   ├── src/
│   │   ├── segmentation.py         # Segmentación (umbralización, watershed, etc.)
│   │   ├── detection.py            # Detección de contornos/componentes
│   │   └── counting.py             # Lógica de conteo y filtrado
│   ├── notebooks/
│   │   └── demo_counting.ipynb
│   ├── data/
│   └── results/
│
├── case3_license_plate/
│   ├── src/
│   │   ├── detection.py            # Detección de la región de la placa
│   │   ├── ocr.py                  # Reconocimiento de texto (EasyOCR/Tesseract)
│   │   └── pipeline.py             # Pipeline completo en tiempo real (video)
│   ├── notebooks/
│   │   └── demo_plate_recognition.ipynb
│   ├── data/
│   └── results/
│
└── tests/                          # Pruebas unitarias (pytest)
```

> Las carpetas `data/raw` y `data/processed` se versionan vacías (con `.gitkeep`); los datasets pesados **no** se suben al repositorio (ver `.gitignore`). Instrucciones de descarga en cada `README.md` interno.

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/image-processing-portfolio.git
cd image-processing-portfolio
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Activar en Linux/macOS
source venv/bin/activate

# Activar en Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Dependencia adicional para OCR (Caso 3)

`pytesseract` requiere el motor **Tesseract-OCR** instalado a nivel de sistema operativo:

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS (Homebrew)
brew install tesseract

# Windows
# Descargar instalador desde:
# https://github.com/UB-Mannheim/tesseract/wiki
```

`easyocr` no requiere instalación adicional a nivel de sistema (descarga sus modelos automáticamente en el primer uso).

---

## ▶️ Ejecución

Cada caso puede ejecutarse de dos formas:

**A. Como script desde línea de comandos:**

```bash
# Caso 1 — Restauración
python case1_restoration/src/restoration.py --input path/to/image.jpg

# Caso 2 — Conteo de objetos
python case2_object_counting/src/counting.py --input path/to/image.jpg

# Caso 3 — Reconocimiento de placas (video en vivo o archivo)
python case3_license_plate/src/pipeline.py --source 0   # webcam
python case3_license_plate/src/pipeline.py --source path/to/video.mp4
```

**B. Como notebook interactivo (recomendado para revisión/demo):**

```bash
jupyter notebook case1_restoration/notebooks/demo_restoration.ipynb
```

---

## 🔬 Metodología general

Los tres casos siguen una estructura metodológica común, adaptada del flujo sugerido en la guía del curso:

1. Definición del problema
2. Obtención/recolección del dataset
3. Análisis exploratorio de las imágenes
4. Preprocesamiento
5. Implementación del pipeline principal (restauración / segmentación+conteo / detección+OCR)
6. Evaluación cuantitativa y cualitativa
7. Discusión de resultados, ventajas y limitaciones

El detalle metodológico específico de cada caso se documenta en su `README.md` interno y en el informe final (`docs/report/`).

---

## 📊 Resultados

Los resultados visuales y cuantitativos de cada caso se almacenan en `resultsX/figures` y `resultsX/metrics` respectivamente. Un resumen consolidado se presenta en el informe final del proyecto (`docs/report/`), junto con el enlace/QR al video explicativo exigido por la guía.

---

## 📦 Dependencias principales

| Librería | Uso |
|---|---|
| OpenCV | Procesamiento de imágenes/video, segmentación, detección de contornos |
| NumPy | Operaciones numéricas y manejo de arreglos |
| scikit-image | Algoritmos adicionales de procesamiento y métricas de calidad |
| Matplotlib / Seaborn | Visualización de resultados |
| Pillow | Manipulación básica de imágenes |
| EasyOCR / PyTesseract | Reconocimiento óptico de caracteres (Caso 3) |
| Jupyter | Notebooks de demostración |

Lista completa en [`requirements.txt`](./requirements.txt).

---

## 🗂️ Datasets sugeridos

| Caso | Datasets |
|---|---|
| 1. Restauración | BSD500, Kodak Lossless Image Dataset, DIV2K, fotografías personales |
| 2. Conteo de objetos | Open Images Dataset, CVPPP Leaf Counting Dataset, datasets de Kaggle, datasets propios |
| 3. Placas vehiculares | CCPD, OpenALPR Benchmark, AOLP, videos autograbados |

> Los datasets no se incluyen en el repositorio por su peso; cada `README.md` interno describe cómo obtenerlos y dónde ubicarlos localmente (`data/raw/`).

---

## 👥 Autores

- [Nombre completo] — [rol / correo institucional]
- [Nombre completo] — [rol / correo institucional] *(si aplica, trabajo en díada)*

---

## 📚 Referencias

- Documentación oficial de [OpenCV](https://docs.opencv.org/)
- Documentación oficial de [scikit-image](https://scikit-image.org/)
- Documentación oficial de [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Referencias bibliográficas y datasets citados según normas [APA/IEEE — ajustar según lineamiento del curso]

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia especificada en [`LICENSE`](./LICENSE). Verificar los términos de uso de los datasets y librerías de terceros referenciados.
