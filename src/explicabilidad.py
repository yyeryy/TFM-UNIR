import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

# Librerías de XAI
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Configurar rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    ParkinsonDataset,
    construir_transformaciones,
    normalizar_roi_frac,
    preparar_dataloaders,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generador de Mapas de Calor (Grad-CAM) para TFM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo .pth del modelo entrenado")
    parser.add_argument("--csv", type=str, default="data_index.csv")
    parser.add_argument("--images", type=str, default="data/PPMI_Procesado_2D_Atlas")
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
    state_dict = checkpoint["model_state_dict"]
    if "fc.1.weight" in state_dict:
        model.fc = nn.Sequential(
            nn.Dropout(p=args_entrenamiento.get("dropout", 0.5)),
            nn.Linear(num_features, num_classes),
        )
    else:
        model.fc = nn.Linear(num_features, num_classes)

    model.load_state_dict(state_dict)
    model = model.to(device)
    # Grad-CAM necesita gradientes en el backbone aunque durante el entrenamiento
    # se haya congelado y solo se haya optimizado la cabeza clasificadora.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()
    
    return model, clases_map, args_entrenamiento

def reproyectar_cam_a_original(grayscale_cam, roi_frac, size=224):
    """
    Lleva el mapa Grad-CAM (calculado sobre la ENTRADA ROI, que es la unica
    entrada valida porque el modelo se entreno con ella) a las coordenadas de
    la imagen ORIGINAL 224x224.

    Como el ROI de entrada es un CenterCrop(crop_size) + Resize(224)
    determinista, su inversa es exacta: se reescala el mapa 224 -> crop_size y
    se coloca en el centro de un lienzo 224x224 a ceros. El calor queda
    confinado a la region central (el modelo nunca recibio informacion fuera
    del recorte), pero se muestra sobre la anatomia completa de la original.
    """
    roi_frac = normalizar_roi_frac(roi_frac)
    crop_size = int(round(size * roi_frac))
    offset = (size - crop_size) // 2

    cam_crop = cv2.resize(
        grayscale_cam.astype(np.float32), (crop_size, crop_size),
        interpolation=cv2.INTER_LINEAR,
    )
    cam_full = np.zeros((size, size), dtype=np.float32)
    cam_full[offset:offset + crop_size, offset:offset + crop_size] = cam_crop
    return cam_full


def desnormalizar_imagen(tensor):
    """
    Revierte la normalización de ImageNet extrayendo los datos directamente
    del transformador oficial de PyTorch.
    """
    transform_oficial = ResNet50_Weights.DEFAULT.transforms()
    
    # Extraemos la media y desviación estandar correctamente
    mean = torch.tensor(transform_oficial.mean).view(3, 1, 1)
    std = torch.tensor(transform_oficial.std).view(3, 1, 1)
    
    tensor_desnorm = tensor.cpu() * std + mean
    return tensor_desnorm

def main():
    args = parse_args()
    device = get_device()
    print(f"[INFO] Dispositivo detectado: {device}")

    ruta_pth = Path(args.checkpoint)
    if not ruta_pth.exists():
        raise FileNotFoundError(f"No se encontro el checkpoint en: {ruta_pth}")
    
    checkpoint = torch.load(ruta_pth, map_location=device, weights_only=False)
    modelo, class_map, args_train = reconstruir_modelo(checkpoint, device)
    
    idx_to_class = {v: k for k, v in class_map.items()}
    clases_permitidas = list(class_map.keys())

    # Reaplicar el mismo ROI de entrada usado en entrenamiento (coherencia train/Grad-CAM)
    roi = args_train.get("roi", True)
    roi_frac = normalizar_roi_frac(args_train.get("roi_frac", 0.6))

    print(f"[INFO] Cargando datos de Test (entrada ROI para el modelo)...")
    _, _, test_loader, _, _ = preparar_dataloaders(
        ruta_csv=PROJECT_ROOT / args.csv,
        ruta_imagenes=PROJECT_ROOT / args.images,
        clases_permitidas=clases_permitidas,
        batch_size=args.num_images,
        roi=roi,
        roi_frac=roi_frac,
    )

    imagenes, etiquetas = next(iter(test_loader))
    imagenes = imagenes.to(device)
    etiquetas = etiquetas.to(device)

    # Recuperar las mismas muestras sin ROI solo para visualizarlas. Reutilizar
    # el dataframe del test evita volver a escanear y dividir todo el dataset.
    imagenes_orig = imagenes
    if roi:
        print(f"[INFO] Cargando imagenes originales 224x224 (sin ROI) para la visualizacion...")
        _, transformacion_original = construir_transformaciones(roi=False)
        dataset_original = ParkinsonDataset(
            test_loader.dataset.df,
            test_loader.dataset.ruta_imagenes,
            transformaciones=transformacion_original,
        )
        test_loader_orig = DataLoader(
            dataset_original,
            batch_size=args.num_images,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        imagenes_orig, _ = next(iter(test_loader_orig))

    target_layers = [modelo.layer4[-1]]
    cam = GradCAM(model=modelo, target_layers=target_layers)

    outputs = modelo(imagenes)
    predicciones = torch.argmax(outputs, dim=1)

    fig, axes = plt.subplots(args.num_images, 2, figsize=(10, 4 * args.num_images))
    if args.num_images == 1:
        axes = [axes] 

    print(f"[INFO] Generando mapas Grad-CAM...")
    
    for i in range(args.num_images):
        tensor_img = imagenes[i:i+1] 
        etiqueta_real = etiquetas[i].item()
        prediccion_red = predicciones[i].item()
        
        nombre_real = idx_to_class[etiqueta_real]
        nombre_pred = idx_to_class[prediccion_red]
        
        target = [ClassifierOutputTarget(prediccion_red)]
        # Grad-CAM SIEMPRE sobre la entrada ROI (unica entrada valida: el
        # modelo se entreno con ella).
        grayscale_cam = cam(input_tensor=tensor_img, targets=target)[0, :]

        if roi:
            # Reproyectar el mapa de calor a coordenadas de la imagen original
            # y mostrarlo sobre la original 224x224 completa.
            cam_para_mostrar = reproyectar_cam_a_original(grayscale_cam, roi_frac)
            img_desnorm = desnormalizar_imagen(imagenes_orig[i])
        else:
            cam_para_mostrar = grayscale_cam
            img_desnorm = desnormalizar_imagen(tensor_img[0])

        img_visual = img_desnorm.numpy().transpose(1, 2, 0)
        img_visual = np.clip(img_visual, 0, 1)

        visualizacion = show_cam_on_image(img_visual, cam_para_mostrar, use_rgb=True)
        
        axes[i][0].imshow(img_visual[:, :, 0], cmap='gray')
        axes[i][0].set_title(f"Real: {nombre_real}", fontsize=12, fontweight='bold')
        axes[i][0].axis('off')
        
        color_titulo = "green" if nombre_real == nombre_pred else "red"
        axes[i][1].imshow(visualizacion)
        axes[i][1].set_title(f"Predicción IA: {nombre_pred}", color=color_titulo, fontsize=12, fontweight='bold')
        axes[i][1].axis('off')

    plt.tight_layout()
    
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ruta_guardado = output_dir / f"gradcam_analisis_{args_train.get('model', 'resnet')}.png"
    
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"[OK] Análisis XAI completado. Imagen guardada en:\n     -> {ruta_guardado}")

if __name__ == "__main__":
    main()
