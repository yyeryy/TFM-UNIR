import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import argparse
import time
from pathlib import Path
import torch

# Ajustar rutas de importación del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.training_loop as training_loop
import src.explicabilidad as explicabilidad
from src.evaluacion import evaluar_modelo_test

def main():
    args = training_loop.parse_args()
    
    print("\n" + "="*80)
    print(f" INICIANDO PIPELINE MLOps: {args.model.upper()}")
    print("="*80)
    
    inicio_pipeline = time.time()
    
    # FASE 1: ENTRENAMIENTO
    print("\n--- FASE 1: BÚSQUEDA Y ENTRENAMIENTO DE PESOS ---")
    training_loop.main()
    
    # BÚSQUEDA DEL ARTEFACTO (Detección del modelo recién creado)
    output_dir = PROJECT_ROOT / args.output_dir
    if args.run_name:
        run_name = args.run_name
    else:
        archivos_existentes = list(output_dir.glob(f"{args.model}_v*_best.pth"))
        if not archivos_existentes:
            print("[ERROR FATAL] La pipeline no pudo encontrar el archivo .pth resultante del entrenamiento.")
            return
        
        # Ordenamos los archivos por tiempo de modificación y seleccionamos el último
        archivos_existentes.sort(key=os.path.getmtime)
        run_name = archivos_existentes[-1].stem.replace("_best", "")

    checkpoint_exacto = output_dir / f"{run_name}_best.pth"
    print(f"\n[PIPELINE] Enlazando automáticamente con el modelo generado: {checkpoint_exacto.name}")
    
    # FASE 2: EVALUACIÓN EN TEST
    print("\n--- FASE 2: EVALUACIÓN MÉDICA EN TEST (DATOS INVISIBLES) ---")
    device = training_loop.get_device(args.device)
    
    test_metrics, _ = evaluar_modelo_test(
        checkpoint_path=checkpoint_exacto,
        csv_path=args.csv,
        images_dir=args.images,
        device=device
    )
    
    print("\n[OK] Fase de evaluacion estatica completada con exito.")
    print(f"     -> Acc Global: {test_metrics['acc']:.4f} | Recall (Sensibilidad): {test_metrics['recall']:.4f} | F1-Score: {test_metrics['f1']:.4f}")
    print(
        f"     -> Paciente: BalAcc {test_metrics.get('patient_balanced_acc', float('nan')):.4f} | "
        f"AUC {test_metrics.get('patient_roc_auc', float('nan')):.4f} | "
        f"Especificidad {test_metrics.get('patient_specificity', float('nan')):.4f} | "
        f"Sensibilidad {test_metrics.get('patient_sensitivity', float('nan')):.4f}"
    )

    # FASE 3: EXPLICABILIDAD (XAI - GRAD-CAM)
    print("\n--- FASE 3: APERTURA DE CAJA NEGRA (XAI) ---")
    
    try:
        # Sobrescribimos temporalmente los argumentos de entrada del sistema (CLI)
        # Esto engaña a explicabilidad.py para que lea el archivo y apunte a la carpeta /XAI
        sys.argv = [
            sys.argv[0], 
            "--checkpoint", str(checkpoint_exacto), 
            "--csv", args.csv,
            "--images", args.images,
            "--output-dir", "XAI"
        ]
        
        # Llamamos directamente a la función principal de tu script XAI
        explicabilidad.main()
        
    except Exception as e:
        print(f"\n[ERROR XAI] Ocurrio un fallo al intentar generar los mapas de calor: {e}")

    # RESUMEN FINAL
    tiempo_total = (time.time() - inicio_pipeline) / 60
    print("\n" + "="*80)
    print(" PIPELINE FINALIZADA CON ÉXITO")
    print(f" Tiempo total de ejecucion: {tiempo_total:.2f} minutos")
    print(f" [1] Modelo .pth guardado en : {checkpoint_exacto}")
    print(f" [2] Registro de metricas en : {args.excel}")
    print(f" [3] Mapas XAI disponibles en: {PROJECT_ROOT / 'XAI'}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
