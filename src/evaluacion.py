import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
import argparse

# Asegurar importaciones relativas desde la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import preparar_dataloaders
from src.training_loop import run_epoch

def get_device():
    """Fuerza la detección del mejor hardware disponible en Mac y Windows"""
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"[INFO] Hardware detectado: NVIDIA GPU ({torch.cuda.get_device_name(0)}) - Usando CUDA")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dispositivo = torch.device("mps")
        print("[INFO] Hardware detectado: Apple Silicon (M1/M2/M3) - Usando MPS")
    else:
        dispositivo = torch.device("cpu")
        print("[WARNING] No se detecto GPU compatible. Usando CPU (Sera muy lento)")
    return dispositivo

def evaluar_modelo_test(checkpoint_path, csv_path="data_index.csv", images_dir="data/PPMI_Procesado_2D", device=None):
    """
    Carga un checkpoint específico y lo evalúa de forma aislada en el conjunto de Test.
    """
    if device is None:
        device = get_device()
    
    checkpoint_path = Path(checkpoint_path)
    print(f"\n[EVALUACIÓN] Cargando modelo desde: {checkpoint_path.name}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"[ERROR] No se encuentra el archivo: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extraer metadatos guardados en el entrenamiento
    args_train = checkpoint.get("args", {})
    model_name = checkpoint.get("model_name", args_train.get("model", "resnet50"))
    class_map = checkpoint.get("class_map", {"Control": 0, "PD": 1})
    clases_permitidas = list(class_map.keys())
    batch_size = args_train.get("batch_size", 32)
    
    # 1. Cargar Dataloader de Test
    _, _, test_loader, class_weights, _ = preparar_dataloaders(
        ruta_csv=PROJECT_ROOT / csv_path,
        ruta_imagenes=PROJECT_ROOT / images_dir,
        clases_permitidas=clases_permitidas,
        batch_size=batch_size
    )
    
    # 2. Reconstruir la arquitectura
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(clases_permitidas))
    
    # Cargar los pesos entrenados y poner en modo evaluación
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 3. Configurar criterio de pérdida
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 4. Ejecutar evaluación en Test
    print(f"[EVALUACIÓN] Procesando lote de Test independiente...")
    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device
    )
    
    return test_metrics, args_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación manual de modelos en el conjunto de Test")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo .pth que quieres evaluar")
    parser.add_argument("--csv", type=str, default="data_index.csv", help="Ruta al archivo de datos CSV")
    parser.add_argument("--images", type=str, default="data/PPMI_Procesado_2D", help="Ruta a las imagenes procesadas")
    args = parser.parse_args()
    
    metrics, _ = evaluar_modelo_test(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        images_dir=args.images
    )
    
    print("\n" + "="*50)
    print(" RESULTADOS FINALES EN TEST (Datos Invisibles)")
    print("="*50)
    
    # Imprimir métricas limpias excluyendo la matriz cruda
    for k, v in metrics.items():
        if k != "confusion_matrix" and v is not None:
            if isinstance(v, float):
                print(f"{k.upper():<15}: {v:.4f}")
            else:
                print(f"{k.upper():<15}: {v}")
    print("="*50)