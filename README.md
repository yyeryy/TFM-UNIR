# TFM-UNIR

## 🛠️ 1. Configuración Inicial (Solo la primera vez)

Si eres el **Usuario 2** o **Usuario 3**, sigue estos pasos para tener el proyecto listo en tu ordenador:

### A. Instalación de herramientas

Abre tu terminal (PowerShell en Windows o Terminal en Mac/Linux) e instala DVC con soporte para Google Drive:
```bash
pip install dvc[gdrive]
```

### B. Clonar el repositorio
```bash
git clone https://github.com/yyeryy/TFM-UNIR.git
cd TFM-UNIR
```

### C. Descargar los datos (El "Check-in" de Drive)

Como el Usuario 1 ya configuró el almacén, tú solo tienes que bajar los archivos reales:
```bash
dvc pull
```

---

## 🔄 2. Flujo de Trabajo Diario (Sincronización)

Para que los tres estemos siempre en la misma página y no haya conflictos, seguid este orden:

### 📥 Al empezar: Bajar cambios

Antes de escribir una sola línea de código, sincroniza tu PC con lo que hayan hecho los demás:
```bash
git pull      # Baja el código nuevo y los "tickets" de DVC
dvc pull      # Baja los archivos pesados (datasets/modelos) que correspondan a ese código
```

### 📤 Al terminar: Subir cambios

Dependiendo de qué hayas tocado, el proceso cambia:

**Escenario 1: Solo has cambiado código (`.py`, `.ipynb`, `.md`)**

Es el proceso normal de GitHub:
```bash
git add .
git commit -m "Explica qué has cambiado"
git push
```

**Escenario 2: Has añadido o cambiado datos/modelos (`.zip`, `.csv`, `.pth`)**

> ⚠️ **IMPORTANTE:** Nunca hagas `git add` de un archivo pesado directamente. Sigue este orden:

1. Registrar en DVC:
```bash
dvc add data/mi_archivo_grande.zip
```

2. Subir el ticket a GitHub:
```bash
git add data/mi_archivo_grande.zip.dvc data/.gitignore
git commit -m "Actualizado dataset/modelo"
git push
```

3. Subir el archivo real a Drive:
```bash
dvc push
```

---

## 🧠 3. Reglas de Oro para el Equipo

- **El "Doble Push":** Si usas DVC, recuerda hacer siempre `git push` (para que los demás vean el ticket) **y** `dvc push` (para que el archivo suba a Drive). Si olvidas el segundo, tus compañeros verán un error al hacer `dvc pull`.
- **No tocar el Drive a mano:** No subas archivos, no borres nada y no cambies nombres dentro de la carpeta de Google Drive desde la web. Deja que DVC lo gestione todo.
- **Estructura de carpetas:**
  - `data/` → Todo lo que sean datos (imágenes, CSV, etc.)
  - `models/` → Todos los pesos de la IA (`.pth`, `.h5`)
  - `src/` → Todo vuestro código ejecutable
- **Conflictos:** Si dos personas tocan el mismo archivo de código a la vez, Git os avisará. Si dos personas tocan el mismo archivo de DVC, la última versión en hacer `dvc push` será la que quede. ¡Comunicación por el grupo de WhatsApp antes de cambios grandes!

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
```
