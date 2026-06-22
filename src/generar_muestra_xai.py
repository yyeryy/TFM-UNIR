import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import ParkinsonDataset
from src.explicabilidad import reconstruir_modelo, desnormalizar_imagen, get_device

SEMILLA = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera una muestra de 150 imágenes Grad-CAM (75 PD + 75 Control) del conjunto de Test."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Ruta al .pth del modelo entrenado (obligatorio).")
    parser.add_argument("--csv", type=str, default="data_index.csv",
                        help="CSV maestro con los metadatos clínicos.")
    parser.add_argument("--images", type=str, default="data/PPMI_Procesado_2D",
                        help="Carpeta con los cortes 2D .npy.")
    parser.add_argument("--output-dir", type=str, default="XAI/muestra-validacion-clinica",
                        help="Carpeta de salida de las imágenes.")
    parser.add_argument("--total", type=int, default=150,
                        help="Número total de imágenes (se reparte 50%% / 50%%).")
    return parser.parse_args()


def fijar_semillas(semilla=SEMILLA):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def obtener_df_test(ruta_csv, ruta_imagenes, clases_permitidas):
    ruta_imagenes = Path(ruta_imagenes)

    df_clinico = pd.read_csv(ruta_csv)
    df_clinico['Subject'] = (
        df_clinico['Subject'].astype(str).str.strip().apply(lambda x: x.split('.')[0])
    )
    df_clinico['Group'] = df_clinico['Group'].astype(str).str.strip()
    df_clinico = df_clinico[df_clinico['Group'].isin(clases_permitidas)]

    df_unico = df_clinico.drop_duplicates(subset=['Subject']).copy()
    mapeo_etiquetas = {clase: idx for idx, clase in enumerate(clases_permitidas)}
    df_unico['Etiqueta'] = df_unico['Group'].map(mapeo_etiquetas)

    archivos_npy = [f.name for f in ruta_imagenes.glob("*.npy")]
    df_archivos = pd.DataFrame({'Archivo': archivos_npy})
    df_archivos['Subject'] = df_archivos['Archivo'].apply(lambda x: x.split('_')[0])

    df_master = pd.merge(df_archivos, df_unico[['Subject', 'Etiqueta']],
                         on='Subject', how='inner')

    sujetos_unicos = df_master[['Subject', 'Etiqueta']].drop_duplicates()

    train_subj, resto_subj = train_test_split(
        sujetos_unicos, test_size=0.30,
        stratify=sujetos_unicos['Etiqueta'], random_state=42
    )
    val_subj, test_subj = train_test_split(
        resto_subj, test_size=0.50,
        stratify=resto_subj['Etiqueta'], random_state=42
    )

    df_test = df_master[df_master['Subject'].isin(test_subj['Subject'])].copy()
    return df_test, mapeo_etiquetas


def muestrear_balanceado(df_test, total, mapeo_etiquetas):
    por_clase = total // 2
    seleccion = []
    for clase, etiqueta in mapeo_etiquetas.items():
        disponibles = df_test[df_test['Etiqueta'] == etiqueta]
        if len(disponibles) < por_clase:
            raise ValueError(
                f"El conjunto de Test solo tiene {len(disponibles)} cortes de la clase "
                f"'{clase}', se necesitan {por_clase}. Reduce --total o cambia el origen."
            )
        seleccion.append(disponibles.sample(n=por_clase, random_state=SEMILLA))

    df_muestra = pd.concat(seleccion, ignore_index=True)
    df_muestra = df_muestra.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    return df_muestra


def main():
    args = parse_args()
    fijar_semillas(SEMILLA)

    device = get_device()
    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] Semilla fija: {SEMILLA}")

    ruta_pth = Path(args.checkpoint)
    if not ruta_pth.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {ruta_pth}")
    checkpoint = torch.load(ruta_pth, map_location=device, weights_only=False)
    modelo, class_map, _ = reconstruir_modelo(checkpoint, device)
    idx_to_class = {v: k for k, v in class_map.items()}
    clases_permitidas = list(class_map.keys())

    ruta_csv = PROJECT_ROOT / args.csv
    ruta_imagenes = PROJECT_ROOT / args.images
    df_test, mapeo_etiquetas = obtener_df_test(ruta_csv, ruta_imagenes, clases_permitidas)
    print(f"[INFO] Cortes en Test: {len(df_test)} "
          f"({df_test['Etiqueta'].value_counts().to_dict()})")

    df_muestra = muestrear_balanceado(df_test, args.total, mapeo_etiquetas)
    print(f"[INFO] Muestra seleccionada: {len(df_muestra)} cortes "
          f"({df_muestra['Etiqueta'].value_counts().to_dict()})")

    dataset = ParkinsonDataset(df_muestra, ruta_imagenes)

    target_layers = [modelo.layer4[-1]]
    cam = GradCAM(model=modelo, target_layers=target_layers)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    registros = []
    print(f"[INFO] Generando {len(df_muestra)} imágenes Grad-CAM...")

    for i in range(len(df_muestra)):
        tensor_img, etiqueta_t = dataset[i]
        tensor_img = tensor_img.unsqueeze(0).to(device)
        etiqueta_real = int(etiqueta_t.item())

        with torch.no_grad():
            salida = modelo(tensor_img)
            prediccion = int(torch.argmax(salida, dim=1).item())

        nombre_real = idx_to_class[etiqueta_real]
        nombre_pred = idx_to_class[prediccion]

        target = [ClassifierOutputTarget(prediccion)]
        grayscale_cam = cam(input_tensor=tensor_img, targets=target)[0, :]

        img_desnorm = desnormalizar_imagen(tensor_img[0])
        img_visual = img_desnorm.numpy().transpose(1, 2, 0)
        img_visual = np.clip(img_visual, 0, 1)
        visualizacion = show_cam_on_image(img_visual, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img_visual[:, :, 0], cmap='gray')
        axes[0].set_title(f"Real: {nombre_real}", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        color_titulo = "green" if nombre_real == nombre_pred else "red"
        axes[1].imshow(visualizacion)
        axes[1].set_title(f"Predicción IA: {nombre_pred}",
                          color=color_titulo, fontsize=11, fontweight='bold')
        axes[1].axis('off')
        plt.tight_layout()

        archivo_npy = df_muestra.loc[i, 'Archivo']
        subject = df_muestra.loc[i, 'Subject']
        base = Path(archivo_npy).stem
        acierto = "OK" if nombre_real == nombre_pred else "FALLO"
        nombre_salida = f"{i + 1:03d}_Real-{nombre_real}_Pred-{nombre_pred}_{base}.png"
        ruta_salida = output_dir / nombre_salida

        plt.savefig(ruta_salida, dpi=200, bbox_inches='tight')
        plt.close(fig)

        registros.append({
            "indice": i + 1,
            "archivo_png": nombre_salida,
            "archivo_npy": archivo_npy,
            "subject": subject,
            "clase_real": nombre_real,
            "prediccion": nombre_pred,
            "acierto": acierto,
        })

        if (i + 1) % 25 == 0:
            print(f"      ... {i + 1}/{len(df_muestra)} imágenes generadas")

    df_indice = pd.DataFrame(registros)
    ruta_indice = output_dir / "indice_muestra.csv"
    df_indice.to_csv(ruta_indice, index=False, encoding="utf-8-sig")

    aciertos = (df_indice['acierto'] == "OK").sum()
    print("\n[OK] Muestra de validación clínica generada.")
    print(f"     -> Imágenes: {output_dir}")
    print(f"     -> Índice  : {ruta_indice}")
    print(f"     -> Aciertos del modelo en la muestra: {aciertos}/{len(df_indice)}")


if __name__ == "__main__":
    main()
