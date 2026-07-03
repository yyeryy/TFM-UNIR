<h1 align="center">Sistema de apoyo al diagnóstico temprano de la Enfermedad de Parkinson mediante Deep Learning y técnicas de Explicabilidad (XAI) en imágenes MRI</h1>

<p align="center">
  <em>Trabajo de Fin de Máster · UNIR · Clasificación Control vs. Parkinson sobre resonancia magnética estructural (MRI-T1) del repositorio PPMI</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/ResNet--50-Transfer%20Learning-blue">
  <img src="https://img.shields.io/badge/Grad--CAM-XAI-2E9E44">
  <img src="https://img.shields.io/badge/Dataset-PPMI-purple">
  <img src="https://img.shields.io/badge/W%26B-tracking-FFBE00?logo=weightsandbiases&logoColor=black">
  <img src="https://img.shields.io/badge/estado-piloto%20experimental-orange">
</p>

---

## Descripción

Este repositorio implementa un **sistema de apoyo al diagnóstico temprano de la Enfermedad de Parkinson** a partir de resonancia magnética estructural (MRI-T1) del repositorio **PPMI**. El código cubre el pipeline completo: preprocesado de las imágenes (alineación al atlas **MNI152**, extracción cerebral y generación de cortes axiales 2D), comparativa y optimización de arquitecturas CNN, entrenamiento de una **ResNet-50** mediante *transfer learning* para clasificar `Control` frente a `Parkinson`, y evaluación con agregación de predicciones por paciente.

El objetivo es doble: por un lado, **detectar patrones morfológicos asociados a la enfermedad** en etapas iniciales; por otro, hacerlo de forma **interpretable** mediante técnicas de explicabilidad (**Grad-CAM**) que permiten auditar las regiones neuroanatómicas en las que se apoya el modelo, incluyendo una inferencia adicional sobre la cohorte *Prodromal*.

---

## Estructura del repositorio

```
TFM-UNIR/
├── 📂 src/                      # Código fuente principal
│   ├── main.py                  # Pipeline MLOps end-to-end
│   ├── dataset.py               # Dataset y DataLoader (split por paciente)
│   ├── training_loop.py         # Bucle de entrenamiento y CLI
│   ├── evaluacion.py            # Evaluación en test (corte y paciente)
│   ├── explicabilidad.py        # Mapas Grad-CAM
│   ├── preprocesado_atlas.py    # Preprocesado con atlas MNI152
│   ├── inferencia_prodromal.py  # Inferencia cohorte Prodromal
│   └── validacion_clinica/      # Atlas y validación de explicaciones
├── 📂 scripts/                  # Comparativa de modelos, Optuna, utilidades
├── 📂 XAI/                      # Salidas de explicabilidad Grad-CAM
├── 📂 models/                   # Checkpoints entrenados (.pth)
├── 📂 resultados/               # Métricas y experimentos (Excel/CSV)
├── 📂 graficas/                 # Figuras y curvas de análisis
├── 📂 notebooks/                # Notebooks de EDA
├── 📄 requirements.txt
└── 📄 README.md
```

---

## Metodología y pipeline

El proyecto sigue un pipeline experimental de 8 fases:

```
1. EDA              →  Perfilado clínico/demográfico y auditoría radiológica
2. Preprocesado     →  Control de calidad · MNI152 · brain extraction · 15 cortes axiales 2D
3. Comparativa      →  5 arquitecturas CNN (ResNet-18/50, DenseNet-121, EfficientNet-B0, MobileNet-V3)
4. Optimización     →  Búsqueda de hiperparámetros con Optuna
5. Arquitectura     →  Definición del modelo final (ResNet-50)
6. Evaluación       →  Métricas en test independiente (por corte y por paciente)
7. Explicabilidad   →  Grad-CAM + validación clínica de las explicaciones
8. Inferencia       →  Evaluación sobre la cohorte Prodromal
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/TFM-UNIR.git
cd TFM-UNIR

# 2. Crear y activar el entorno virtual (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

**Pipeline completo** (entrenamiento → evaluación en test → generación de mapas XAI):

```bash
python -m src.main --model resnet50
```

**Entrenamiento con hiperparámetros personalizados:**

```bash
python -m src.training_loop --model resnet50 --epochs 60 --batch-size 32 \
    --lr 5e-5 --dropout 0.5 --freeze layer4 --amp
```

**Otras utilidades:**

```bash
python src/preprocesado_atlas.py        # Preprocesado MNI152 + extracción de cortes 2D
python scripts/comparativa_modelos.py   # Comparativa de las 5 arquitecturas CNN
python scripts/optuna_search.py         # Optimización de hiperparámetros
python src/explicabilidad.py            # Generación de mapas Grad-CAM
python src/inferencia_prodromal.py      # Inferencia sobre la cohorte Prodromal
```

---
