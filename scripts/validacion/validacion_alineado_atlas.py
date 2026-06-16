import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# 1. DEFINICIÓN DE RUTAS (Idénticas al script principal)
DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_DATOS = DIRECTORIO_RAIZ / "data" / "PPMI_Procesado_2D_Atlas"

def ejecutar_auditoria(num_muestras=4, corte_a_revisar="07"):
    print("="*60)
    print(f" SCRIPT DE AUDITORÍA: COMPROBACIÓN DEL CORTE AXIAL {corte_a_revisar}")
    print("="*60)
    
    if not RUTA_DATOS.exists():
        print(f"[ERROR] No se encuentra la carpeta de salida: {RUTA_DATOS}")
        print("Asegúrate de que el script principal ya haya procesado al menos a unos pocos pacientes.")
        return

    # Buscamos todos los archivos .npy que correspondan al MISMO corte exacto (ej. corte 07)
    # Esto nos permite validar si la alineación espacial es idéntica entre sujetos
    archivos_corte = list(RUTA_DATOS.glob(f"*_axial_{corte_a_revisar}.npy"))
    
    total_disponibles = len(archivos_corte)
    print(f"[INFO] Pacientes procesados disponibles para auditar: {total_disponibles}")
    
    if total_disponibles < num_muestras:
        print(f"[AVISO] Hay pocos datos aún ({total_disponibles}). Mostrando los que hay...")
        num_muestras = total_disponibles
        
    if total_disponibles == 0:
        print("[INFO] Esperando a que el script principal genere los primeros archivos...")
        return

    # Selección aleatoria sin repetición
    muestras = random.sample(archivos_corte, num_muestras)

    # Configurar el lienzo de Matplotlib
    fig, axes = plt.subplots(1, num_muestras, figsize=(4 * num_muestras, 4))
    if num_muestras == 1:
        axes = [axes] # Forzar lista si solo hay un gráfico
        
    plt.suptitle(f"Auditoría MNI152 + Skull Stripping - Corte Fijo {corte_a_revisar}", 
                 fontweight='bold', fontsize=14, y=1.05)

    print("\n--- MÉTRICAS DE LOS TENSORES EVALUADOS ---")
    
    for i, archivo in enumerate(muestras):
        # Cargar la matriz matemática
        img = np.load(archivo)
        shape = img.shape
        min_val, max_val = img.min(), img.max()
        
        # Extraer ID del sujeto del nombre del archivo
        id_sujeto = archivo.name.split('_')[0]
        print(f"Sujeto: {id_sujeto} | Matriz: {shape} | Rango Píxeles: [{min_val:.2f}, {max_val:.2f}]")
        
        # Dibujar el cerebro en escala de grises real
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Sujeto: {id_sujeto}", fontsize=11, fontweight='semibold')
        
        # Dibujar líneas guía rojas en el centro exacto (112, 112)
        axes[i].axhline(y=shape[0]//2, color='red', linestyle='--', alpha=0.4, linewidth=1)
        axes[i].axvline(x=shape[1]//2, color='red', linestyle='--', alpha=0.4, linewidth=1)
        axes[i].axis('off')

    print("-" * 42)
    plt.tight_layout()
    print("\n[INFO] Desplegando ventana gráfica. Cérrala para finalizar el script.")
    plt.show()

if __name__ == "__main__":
    # Puedes cambiar el 'corte_a_revisar' por cualquier número del "00" al "14"
    ejecutar_auditoria(num_muestras=4, corte_a_revisar="07")