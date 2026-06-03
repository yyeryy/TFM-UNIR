import os
import sys
import glob
import pandas as pd

# Configuracion de rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir)) 

dir_indexs = os.path.join(project_root, 'data', 'indexs')
dir_metadata = os.path.join(project_root, 'data', 'metadata')
dir_ppmi_raw = os.path.join(project_root, 'data', 'PPMI')

csv_pd_control = os.path.join(dir_indexs, 'ppmi_pd_control.csv')
csv_prodromal = os.path.join(dir_indexs, 'ppmi_prodromal.csv')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Directorio creado: {path}")

def load_csv(path):
    if not os.path.exists(path):
        print(f"[ERROR] Archivo no encontrado: {path}")
        return None
    try:
        df = pd.read_csv(path)
        if 'Subject' not in df.columns:
            print(f"[ERROR] Columna 'Subject' faltante en {path}")
            return None
        return df
    except Exception as e:
        print(f"[ERROR] Fallo al leer {path}: {e}")
        return None

def main():
    print("Iniciando validacion de datos crudos...")
    ensure_dir(dir_indexs)
    ensure_dir(dir_metadata)
    ensure_dir(dir_ppmi_raw)

    status_ok = True

    # 1. Verificacion de CSV
    print("\n[1/4] Comprobando archivos de metadatos CSV...")
    df_pd = load_csv(csv_pd_control)
    df_prod = load_csv(csv_prodromal)

    if df_pd is None or df_prod is None:
        print("[FATAL] Faltan archivos CSV base. Ejecucion detenida.")
        sys.exit(1)

    print(f"[OK] Cargados {len(df_pd)} registros PD/Control y {len(df_prod)} registros Prodromal.")

    subjects_pd = set(df_pd['Subject'].astype(str).str.strip())
    subjects_prod = set(df_prod['Subject'].astype(str).str.strip())
    all_subjects = subjects_pd.union(subjects_prod)

    # 2. Verificacion de cruce de datos
    print("\n[2/4] Verificando interseccion de cohortes (Data Leakage)...")
    intersection = subjects_pd.intersection(subjects_prod)
    
    if len(intersection) == 0:
        print("[OK] Cero sujetos solapados entre cohortes.")
    else:
        print(f"[ERROR] Se detectaron {len(intersection)} sujetos repetidos: {intersection}")
        status_ok = False

    # 3. Verificacion de directorios de imagenes
    print("\n[3/4] Comprobando directorios fisicos de imagenes...")
    missing_dirs = []
    
    for subject_id in all_subjects:
        subject_dir = os.path.join(dir_ppmi_raw, subject_id)
        if not os.path.exists(subject_dir) or not os.listdir(subject_dir):
            missing_dirs.append(subject_id)

    if len(missing_dirs) == 0:
        print(f"[OK] Los {len(all_subjects)} directorios de sujetos estan presentes.")
    else:
        print(f"[ERROR] Faltan directorios de imagenes para {len(missing_dirs)} sujetos.")
        print("[INFO] Accion requerida: Descargar las imagenes crudas en data/PPMI/")
        status_ok = False

    # 4. Verificacion de XML
    print("\n[4/4] Comprobando archivos XML de metadatos...")
    missing_xmls = 0
    
    for subject_id in all_subjects:
        if not glob.glob(os.path.join(dir_metadata, f"*{subject_id}*.xml")):
            missing_xmls += 1

    if missing_xmls == 0:
        print("[OK] Todos los archivos XML estan presentes.")
    else:
        print(f"[WARNING] Faltan metadatos XML para {missing_xmls} sujetos. (Proceso no bloqueante)")

    # Resultado
    print("\nResumen de validacion:")
    if status_ok:
        print("[ESTADO] PASSED. Datos listos para preprocesamiento.")
        sys.exit(0)
    else:
        print("[ESTADO] FAILED. Se requiere corregir los errores listados.")
        sys.exit(1)

if __name__ == "__main__":
    main()