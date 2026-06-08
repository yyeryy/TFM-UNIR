import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models

# Librerías de XAI
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configurar rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import preparar_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Generador de Mapas de Calor (Grad-CAM) para TFM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo .pth del modelo entrenado")
    parser.add_argument("--csv", type=str, default="data_index.csv")
    parser.add_argument("--images", type=str, default="data/PPMI_Procesado_2D")
    parser.add_argument("--output-dir", type=str, default="graficas")
    parser.add_argument("--num-images", type=int, default=4, help="Numero de imagenes a analizar y graficar")
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

def reconstruir_modelo(checkpoint, device):
    """Reconstruye la arquitectura dinámicamente leyendo los metadatos del .pth"""
    args_entrenamiento = checkpoint.get("args", {})
    model_name = checkpoint.get("model_name", args_entrenamiento.get("model", "resnet50"))
    clases_map = checkpoint.get("class_map", {"Control": 0, "PD": 1})
    num_classes = len(clases_map)

    print(f"[INFO] Reconstruyendo {model_name} para {num_classes} clases...")
    
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Modelo {model_name} no soportado en este script.")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # Inyectar pesos y poner en modo evaluación (Crítico para XAI)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, clases_map, args_entrenamiento

def main():
    args = parse_args()
    device = get_device()
    print(f"[INFO] Dispositivo detectado: {device}")

    # 1. Cargar el Checkpoint
    ruta_pth = PROJECT_ROOT / args.checkpoint
    if not ruta_pth.exists():
        raise FileNotFoundError(f"No se encontro el checkpoint en: {ruta_pth}")
    
    checkpoint = torch.load(ruta_pth, map_location=device)
    modelo, class_map, args_train = reconstruir_modelo(checkpoint, device)
    
    # Invertir el diccionario para mapear de numero a nombre de clase
    idx_to_class = {v: k for k, v in class_map.items()}
    clases_permitidas = list(class_map.keys())

    # 2. Cargar Dataloader (Solo necesitamos Test)
    print(f"[INFO] Cargando datos de Test...")
    _, _, test_loader, _, _ = preparar_dataloaders(
        ruta_csv=PROJECT_ROOT / args.csv,
        ruta_imagenes=PROJECT_ROOT / args.images,
        clases_permitidas=clases_permitidas,
        batch_size=args.num_images, # Cargamos justo las imagenes que queremos mostrar
    )

    # Extraer un lote
    imagenes, etiquetas = next(iter(test_loader))
    imagenes = imagenes.to(device)
    etiquetas = etiquetas.to(device)

    # 3. Configurar Grad-CAM
    # En ResNet, la última capa convolucional útil es la layer4
    target_layers = [modelo.layer4[-1]]
    cam = GradCAM(model=modelo, target_layers=target_layers)

    # 4. Generar predicciones y mapas
    outputs = modelo(imagenes)
    predicciones = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(args.num_images, 2, figsize=(10, 4 * args.num_images))
    if args.num_images == 1:
        axes = [axes] # Asegurar que sea iterable si solo pedimos 1 imagen

    print(f"[INFO] Generando mapas Grad-CAM...")
    
    for i in range(args.num_images):
        tensor_img = imagenes[i:i+1] # Mantener dimension de batch [1, C, H, W]
        etiqueta_real = etiquetas[i].item()
        prediccion_red = predicciones[i].item()
        
        nombre_real = idx_to_class[etiqueta_real]
        nombre_pred = idx_to_class[prediccion_red]
        
        # Le pedimos a Grad-CAM que explique por qué predijo lo que predijo
        target = [ClassifierOutputTarget(prediccion_red)]
        
        # Generar máscara térmica [H, W]
        grayscale_cam = cam(input_tensor=tensor_img, targets=target)[0, :]
        
        # Procesar imagen original para visualización
        img_original = tensor_img[0].cpu().numpy().transpose(1, 2, 0) # [H, W, 3]
        
        # Normalizar min-max estrictamente entre 0 y 1 para que show_cam_on_image no falle
        img_norm = (img_original - img_original.min()) / (img_original.max() - img_original.min() + 1e-8)
        
        # Fusión térmica
        visualizacion = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)
        
        # Columna 1: Original
        axes[i][0].imshow(img_norm[:, :, 0], cmap='gray')
        axes[i][0].set_title(f"Real: {nombre_real}", fontsize=12, fontweight='bold')
        axes[i][0].axis('off')
        
        # Columna 2: Grad-CAM
        color_titulo = "green" if nombre_real == nombre_pred else "red"
        axes[i][1].imshow(visualizacion)
        axes[i][1].set_title(f"Predicción IA: {nombre_pred}", color=color_titulo, fontsize=12, fontweight='bold')
        axes[i][1].axis('off')

    plt.tight_layout()
    
    # 5. Guardar resultado
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ruta_guardado = output_dir / f"gradcam_analisis_{args_train.get('model', 'resnet')}.png"
    
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"[OK] Análisis XAI completado. Imagen guardada en:\n     -> {ruta_guardado}")

if __name__ == "__main__":
    main()