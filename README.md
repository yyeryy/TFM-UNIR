# TFM-UNIR


Breve descripción del proyecto (1-2 frases). Ejemplo: *Implementación de un modelo de segmentación de imágenes médicas utilizando arquitecturas basadas en Transformers para la detección temprana de anomalías.*

---

## 📂 Estructura del Repositorio

Hemos organizado el proyecto siguiendo un esquema modular para facilitar el trabajo en paralelo y la escalabilidad del código:

```text
.
├── data/               # Datasets originales y procesados (ignorado por Git)
├── notebooks/          # Experimentos iniciales, EDA y prototipado (.ipynb)
├── src/                # Código fuente definitivo y modular (.py)
│   ├── preprocessing/  # Limpieza, normalización y aumento de datos
│   ├── models/         # Definición de arquitecturas (PyTorch/TF)
│   └── training/       # Scripts de entrenamiento y validación
├── models/             # Pesos de modelos entrenados (.pth, .h5, .onnx)
├── tests/              # Pruebas para asegurar la integridad del pipeline
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación principal
