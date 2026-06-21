import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import normalizar_roi_frac, preparar_dataloaders
from src.training_loop import run_epoch

def get_device():
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

def evaluar_modelo_test(checkpoint_path, csv_path="data_index.csv", images_dir="data/PPMI_Procesado_2D_Atlas", device=None):
    if device is None:
        device = get_device()
    
    checkpoint_path = Path(checkpoint_path)
    print(f"\n[EVALUACIÓN] Cargando modelo desde: {checkpoint_path.name}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"[ERROR] No se encuentra el archivo: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    args_train = checkpoint.get("args", {})
    model_name = checkpoint.get("model_name", args_train.get("model", "resnet50"))
    class_map = checkpoint.get("class_map", {"Control": 0, "PD": 1})
    clases_permitidas = list(class_map.keys())
    batch_size = args_train.get("batch_size", 32)

    roi = args_train.get("roi", True)
    roi_frac = normalizar_roi_frac(args_train.get("roi_frac", 0.6))
    balance_strategy = args_train.get("balance_strategy", "class_weights")

    _, _, test_loader, class_weights, _ = preparar_dataloaders(
        ruta_csv=PROJECT_ROOT / csv_path,
        ruta_imagenes=PROJECT_ROOT / images_dir,
        clases_permitidas=clases_permitidas,
        batch_size=batch_size,
        roi=roi,
        roi_frac=roi_frac,
        balance_strategy=balance_strategy,
        return_subject=True,
    )
    
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
        
    num_features = model.fc.in_features
    state_dict = checkpoint["model_state_dict"]
    if "fc.1.weight" in state_dict:
        model.fc = nn.Sequential(
            nn.Dropout(p=args_train.get("dropout", 0.5)),
            nn.Linear(num_features, len(clases_permitidas)),
        )
    else:
        model.fc = nn.Linear(num_features, len(clases_permitidas))
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args_train.get("label_smoothing", 0.0),
    )
    
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
    parser.add_argument("--images", type=str, default="data/PPMI_Procesado_2D_Atlas", help="Ruta a las imagenes procesadas")
    args = parser.parse_args()
    
    metrics, _ = evaluar_modelo_test(
        checkpoint_path=args.checkpoint,
        csv_path=args.csv,
        images_dir=args.images
    )
    
    print("\n" + "="*50)
    print(" RESULTADOS FINALES EN TEST (Datos Invisibles)")
    print("="*50)
    
    for k, v in metrics.items():
        if k != "confusion_matrix" and v is not None:
            if isinstance(v, float):
                print(f"{k.upper():<15}: {v:.4f}")
            else:
                print(f"{k.upper():<15}: {v}")
    print("="*50)
