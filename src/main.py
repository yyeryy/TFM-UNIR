import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

RUTA_DATOS_PROCESADOS = Path("../data/PPMI_Procesado")
RUTA_CSV_CLINICO = Path("../data_index.csv")

TAMANO_LOTE = 32
EPOCAS = 30
TASA_APRENDIZAJE = 1e-4

def main():
    #PASO 1: Preparacion de Datos y Etiquetas
    #PASO 2: Construccion de DataLoaders
    #PASO 3: Arquitectura del Modelo
    #PASO 4: Funcion de Perdida y Optimizador
    #PASO 5: Bucle de Entrenamiento
    #PASO 6: Evaluacion y Explicabilidad (XAI)
    print("Pendiente")
    
if __name__ == "__main__":
    main()