import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importamos las herramientas de tus otros scripts
from src.dataset import ParkinsonDataset, construir_transformaciones, normalizar_roi_frac
from src.training_loop import build_model, get_device

def evaluar_cohorte(modelo, loader, device):
    """Pasa una cohorte por el modelo y devuelve un DataFrame con las probabilidades por paciente"""
    resultados = []
    with torch.no_grad():
        for batch in loader:
            # Tu Dataset modificado devuelve 3 cosas: images, labels, subjects
            if len(batch) == 3:
                images, _, subjects = batch
            else:
                images, labels = batch
                subjects = [f"Subj_Desconocido_{i}" for i in range(len(images))]

            images = images.to(device)
            outputs = modelo(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() # Probabilidad de PD
            
            for p, subject in zip(probs, subjects):
                resultados.append({"Subject": subject, "Probabilidad_PD_Corte": float(p)})
                
    df = pd.DataFrame(resultados)
    # Agrupamos haciendo la media por paciente
    return df.groupby("Subject").agg(Probabilidad_Media_PD=("Probabilidad_PD_Corte", "mean")).reset_index()

def cargar_grupo(ruta_csv, ruta_imagenes, grupo_nombre, roi, roi_frac):
    """Carga un DataLoader específico aislando a un solo grupo clínico"""
    df = pd.read_csv(ruta_csv)
    df['Subject'] = df['Subject'].astype(str).str.strip().apply(lambda x: x.split('.')[0])
    df['Group'] = df['Group'].astype(str).str.strip()
    
    df_grupo = df[df['Group'] == grupo_nombre].copy()
    if len(df_grupo) == 0: 
        return None
    
    df_grupo['Etiqueta'] = 0 # Dummy, no la usamos para inferencia
    df_unico = df_grupo.drop_duplicates(subset=['Subject']).copy()
    
    archivos_npy = [f.name for f in Path(ruta_imagenes).glob("*.npy")]
    df_archivos = pd.DataFrame({'Archivo': archivos_npy})
    df_archivos['Subject'] = df_archivos['Archivo'].apply(lambda x: x.split('_')[0])
    
    df_master = pd.merge(df_archivos, df_unico[['Subject', 'Etiqueta']], on='Subject', how='inner')
    
    _, eval_transforms = construir_transformaciones(roi=roi, roi_frac=roi_frac)
    dataset = ParkinsonDataset(df_master, ruta_imagenes, transformaciones=eval_transforms, return_subject=True)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

def main():
    # 1. Configuración de rutas
    device = get_device("auto")
    checkpoint_path = PROJECT_ROOT / "models" / "resnet50_best_definitive_best.pth"
    csv_path = PROJECT_ROOT / "data_index.csv"
    img_path = PROJECT_ROOT / "data" / "PPMI_Procesado_2D_Atlas"
    
    print("\n" + "="*80)
    print(" ANÁLISIS ESTADÍSTICO DE TRAYECTORIA DE LA ENFERMEDAD (TFM)")
    print("="*80)
    
    if not checkpoint_path.exists():
        print(f"[ERROR] No se encuentra el modelo en: {checkpoint_path}")
        return

    # 2. Cargar Checkpoint y Reconstruir Modelo
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_train = checkpoint.get("args", {})
    roi = args_train.get("roi", True)
    roi_frac = normalizar_roi_frac(args_train.get("roi_frac", 0.6))
    dropout = args_train.get("dropout", 0.5)
    
    # Reconstruimos la ResNet50 tal cual se entrenó
    model = build_model(model_name="resnet50", num_classes=2, pretrained=False, freeze="none", dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 3. Extraer probabilidades de los 3 grupos clínicos
    print("[INFO] Evaluando Grupo Control...")
    loader_control = cargar_grupo(csv_path, img_path, "Control", roi, roi_frac)
    df_control = evaluar_cohorte(model, loader_control, device)
    df_control["Clase_Clinica"] = "Control (Sano)"

    print("[INFO] Evaluando Grupo Prodromal...")
    loader_prodromal = cargar_grupo(csv_path, img_path, "Prodromal", roi, roi_frac)
    df_prodromal = evaluar_cohorte(model, loader_prodromal, device)
    df_prodromal["Clase_Clinica"] = "Prodromal (Transición)"

    print("[INFO] Evaluando Grupo PD...")
    loader_pd = cargar_grupo(csv_path, img_path, "PD", roi, roi_frac)
    df_pd = evaluar_cohorte(model, loader_pd, device)
    df_pd["Clase_Clinica"] = "Parkinson (PD)"

    # Combinar en un solo DataFrame maestro
    df_total = pd.concat([df_control, df_prodromal, df_pd], ignore_index=True)

    # 4. Análisis Estadístico Descriptivo y Kruskal-Wallis
    print("\n[RESULTADOS ESTADÍSTICOS]")
    print(df_total.groupby("Clase_Clinica")["Probabilidad_Media_PD"].describe()[['count', 'mean', 'std']])
    
    h_stat, p_valor = stats.kruskal(
        df_control["Probabilidad_Media_PD"], 
        df_prodromal["Probabilidad_Media_PD"], 
        df_pd["Probabilidad_Media_PD"]
    )
    
    print("\n" + "-"*40)
    print(" PRUEBA DE KRUSKAL-WALLIS")
    print("-"*40)
    print(f"Estadístico H : {h_stat:.4f}")
    print(f"P-Valor       : {p_valor:.4e}")
    if p_valor < 0.05:
        print("CONCLUSIÓN    : Las tres etapas de la enfermedad son estadísticamente distinguibles por la IA (p < 0.05).")
    else:
        print("CONCLUSIÓN    : No hay evidencia estadística suficiente para separar los tres grupos.")

    # 5. Generar Gráfico Científico
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(10, 6))
    
    colores = {"Control (Sano)": "#2ecc71", "Prodromal (Transición)": "#f1c40f", "Parkinson (PD)": "#e74c3c"}
    
    sns.kdeplot(
        data=df_total, x="Probabilidad_Media_PD", hue="Clase_Clinica",
        fill=True, common_norm=False, palette=colores, alpha=0.5, linewidth=2
    )
    
    plt.title("Distribución de Probabilidad de Parkinson según Etapa Clínica", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Probabilidad asignada por la ResNet-50 (P(PD))", fontsize=12)
    plt.ylabel("Densidad de Pacientes", fontsize=12)
    plt.xlim(-0.1, 1.1)
    
    out_dir = PROJECT_ROOT / "resultados"
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_grafico = out_dir / "distribucion_continua_enfermedad.png"
    plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
    
    print(f"\n[OK] Gráfico científico guardado en: {ruta_grafico}")

if __name__ == "__main__":
    main()