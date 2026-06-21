import sys
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent

dir_procesado_2d = project_root / 'data' / 'PPMI_Procesado_2D'
archivo_maestro = project_root / 'data_index.csv'

NUMERO_CORTES = 15

def main():
    print("Iniciando auditoria multiplataforma de imagenes preprocesadas (Mac/Windows)...")
    status_ok = True

    if not archivo_maestro.exists():
        print(f"[FATAL] No se encuentra {archivo_maestro.name}.")
        sys.exit(1)
        
    if not dir_procesado_2d.exists():
        print(f"[FATAL] No existe el directorio de imagenes: {dir_procesado_2d}")
        sys.exit(1)

    try:
        df_maestro = pd.read_csv(archivo_maestro)
        df_maestro['Subject'] = df_maestro['Subject'].astype(str).str.strip().apply(lambda x: x.split('.')[0]) 
               
        subjects = set(df_maestro['Subject'])
        total_pacientes = len(subjects)
        total_archivos_esperados = total_pacientes * NUMERO_CORTES
    except Exception as e:
        print(f"[ERROR] No se pudo leer {archivo_maestro.name}: {e}")
        sys.exit(1)

    print(f"\n[1/3] Verificando completitud de cortes por paciente...")
    pacientes_incompletos = 0
    archivos_faltantes = 0
    
    for subject_id in subjects:
        faltan_en_sujeto = 0
        for i in range(NUMERO_CORTES):
            nombre_archivo = f"{subject_id}_axial_{i:02d}.npy"
            ruta_archivo = dir_procesado_2d / nombre_archivo
            
            if not ruta_archivo.is_file():
                faltan_en_sujeto += 1
                archivos_faltantes += 1
        
        if faltan_en_sujeto > 0:
            pacientes_incompletos += 1

    if pacientes_incompletos == 0:
        print(f"[OK] Los {total_pacientes} pacientes tienen sus {NUMERO_CORTES} cortes generados.")
    else:
        print(f"[ERROR] {pacientes_incompletos} pacientes incompletos (faltan {archivos_faltantes} cortes).")
        status_ok = False

    print(f"\n[2/3] Verificando archivos fantasma (Limpieza del directorio)...")

    archivos_reales = list(dir_procesado_2d.glob('*.npy'))
    total_archivos_reales = len(archivos_reales)
    
    print(f"[INFO] Archivos .npy detectados en disco: {total_archivos_reales}")
    print(f"[INFO] Archivos .npy esperados segun indice: {total_archivos_esperados}")
    
    if total_archivos_reales == total_archivos_esperados:
        print("[OK] El directorio esta limpio. No hay archivos residuales.")
    else:
        archivos_fantasma = total_archivos_reales - total_archivos_esperados
        if archivos_fantasma > 0:
            print(f"[ERROR] Tienes {archivos_fantasma} archivos extra que no constan en el indice.")
        else:
            print(f"[ERROR] Faltan {abs(archivos_fantasma)} archivos fisicos en la carpeta.")
        status_ok = False

    print(f"\n[3/3] Verificando integridad binaria (File Size Validation)...")
    if total_archivos_reales > 0:

        tamano_referencia = archivos_reales[0].stat().st_size
        archivos_corruptos = sum(1 for f in archivos_reales if f.stat().st_size != tamano_referencia)
                
        if archivos_corruptos == 0:
            print(f"[OK] El 100% de las imagenes pesan exactamente {tamano_referencia} bytes.")
        else:
            print(f"[ERROR] Se encontraron {archivos_corruptos} archivos con un tamano distinto al esperado.")
            status_ok = False
    else:
        print("[WARNING] No se encontraron archivos para verificar el tamano.")

    print("\nResumen de validacion de imagenes:")
    if status_ok:
        print("[ESTADO] PASSED. Dataset 2D en perfecto estado multiplataforma.")
        sys.exit(0)
    else:
        print("[ESTADO] FAILED. Corrige los errores en la generacion de imagenes.")
        sys.exit(1)

if __name__ == "__main__":
    main()