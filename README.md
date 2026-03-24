# TFM-UNIR

Estructura de Carpetas Sugerida (IA)

Plaintext
├── data/               # Datasets (añadid esto al .gitignore para no subirlos)
├── notebooks/          # Experimentos iniciales en Jupyter (.ipynb)
├── src/                # Código fuente definitivo (.py)
│   ├── preprocessing/  # Limpieza de datos
│   ├── models/         # Definición de la arquitectura
│   └── training/       # Scripts de entrenamiento
├── models/             # Pesos guardados (.pth, .h5) -> Usad DVC o Git LFS
├── tests/              # Pruebas para asegurar que el código funciona
├── requirements.txt    # Librerías necesarias (pip install -r ...)
└── README.md           # Documentación del proyecto
