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
archivo_maestro = os.path.join(project_root, 'data_index.csv')

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
    print("Iniciando validacion de datos del proyecto...")
    ensure_dir(dir_indexs)
    ensure_dir(dir_metadata)
    ensure_dir(dir_ppmi_raw)

    status_ok = True

    # 1. Verificacion de CSV crudos
    print("\n[1/5] Comprobando archivos de metadatos CSV...")
    df_pd = load_csv(csv_pd_control)
    df_prod = load_csv(csv_prodromal)

    if df_pd is None or df_prod is None:
        print("[FATAL] Faltan archivos CSV base. Ejecucion detenida.")
        sys.exit(1)

    print(f"[OK] Cargados {len(df_pd)} registros PD/Control y {len(df_prod)} registros Prodromal.")

    subjects_pd = set(df_pd['Subject'].astype(str).str.strip())
    subjects_prod = set(df_prod['Subject'].astype(str).str.strip())
    all_subjects_raw = subjects_pd.union(subjects_prod)

    # 2. Verificacion de cruce de datos
    print("\n[2/5] Verificando interseccion de cohortes (Data Leakage)...")
    intersection = subjects_pd.intersection(subjects_prod)
    
    if len(intersection) == 0:
        print("[OK] Cero sujetos solapados entre cohortes.")
    else:
        print(f"[ERROR] Se detectaron {len(intersection)} sujetos repetidos: {intersection}")
        status_ok = False

    # 3. Verificacion de directorios de imagenes
    print("\n[3/5] Comprobando directorios fisicos de imagenes...")
    missing_dirs = []
    
    for subject_id in all_subjects_raw:
        subject_dir = os.path.join(dir_ppmi_raw, subject_id)
        if not os.path.exists(subject_dir) or not os.listdir(subject_dir):
            missing_dirs.append(subject_id)

    if len(missing_dirs) == 0:
        print(f"[OK] Los {len(all_subjects_raw)} directorios de sujetos estan presentes.")
    else:
        print(f"[ERROR] Faltan directorios de imagenes para {len(missing_dirs)} sujetos.")
        print("[INFO] Accion requerida: Descargar las imagenes crudas en data/PPMI/")
        status_ok = False

    # 4. Verificacion de XML
    print("\n[4/5] Comprobando archivos XML de metadatos...")
    missing_xmls = 0
    
    for subject_id in all_subjects_raw:
        if not glob.glob(os.path.join(dir_metadata, f"*{subject_id}*.xml")):
            missing_xmls += 1

    if missing_xmls == 0:
        print("[OK] Todos los archivos XML estan presentes.")
    else:
        print(f"[WARNING] Faltan metadatos XML para {missing_xmls} sujetos. (Proceso no bloqueante)")

    # 5. Verificacion del Indice Maestro
    print("\n[5/5] Comprobando el Indice Maestro (data_index.csv)...")
    if not os.path.exists(archivo_maestro):
        print("[ERROR] El archivo data_index.csv no existe en la raiz del proyecto.")
        print("[INFO] Accion requerida: Ejecutar 'python scripts/generar_indice.py'")
        status_ok = False
    else:
        try:
            df_maestro = pd.read_csv(archivo_maestro)
            if 'subject_id' not in df_maestro.columns:
                print("[ERROR] data_index.csv no tiene la columna 'subject_id'. Formato invalido.")
                status_ok = False
            else:
                subjects_maestro = set(df_maestro['subject_id'].astype(str).str.strip())
                
                faltan_en_maestro = all_subjects_raw - subjects_maestro
                sobran_en_maestro = subjects_maestro - all_subjects_raw
                
                if len(faltan_en_maestro) == 0 and len(sobran_en_maestro) == 0:
                    print(f"[OK] Indice maestro perfectamente sincronizado ({len(subjects_maestro)} pacientes).")
                else:
                    if len(faltan_en_maestro) > 0:
                        print(f"[ERROR] Faltan {len(faltan_en_maestro)} sujetos en data_index.csv.")
                    if len(sobran_en_maestro) > 0:
                        print(f"[ERROR] Hay {len(sobran_en_maestro)} sujetos fantasma en data_index.csv que no estan en los datos crudos.")
                    print("[INFO] Accion requerida: Reconstruir el indice ejecutando 'python scripts/generar_indice.py'")
                    status_ok = False
        except Exception as e:
            print(f"[ERROR] Fallo al leer data_index.csv: {e}")
            status_ok = False

    # Resultado
    print("\nResumen de validacion:")
    if status_ok:
        print("[ESTADO] PASSED. Pipeline de datos completo e integro.")
        sys.exit(0)
    else:
        print("[ESTADO] FAILED. Se requiere corregir los errores listados.")
        sys.exit(1)

if __name__ == "__main__":
    main()