import pandas as pd
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
archivo_maestro = os.path.join(project_root, 'data_index.csv')
dir_procesado_2d = os.path.join(project_root, 'data', 'PPMI_Procesado_2D')
NUMERO_CORTES = 15

def main():
    print("Iniciando purga de pacientes defectuosos del índice...")

    if not os.path.exists(archivo_maestro):
        print("[ERROR] No existe data_index.csv")
        sys.exit(1)

    df_maestro = pd.read_csv(archivo_maestro)
    total_original = len(df_maestro)
    
    df_maestro['Subject'] = df_maestro['Subject'].astype(str).str.strip()

    indices_a_mantener = []
    pacientes_eliminados = []

    print("Verificando existencia física de imágenes paciente por paciente...")
    
    for index, row in df_maestro.iterrows():
        subject_id = row['Subject']
        es_valido = True
        
        for i in range(NUMERO_CORTES):
            nombre_archivo = f"{subject_id}_axial_{i:02d}.npy"
            ruta = os.path.join(dir_procesado_2d, nombre_archivo)
            if not os.path.exists(ruta):
                es_valido = False
                break # Si falta uno, descartamos al paciente entero
                
        if es_valido:
            indices_a_mantener.append(index)
        else:
            pacientes_eliminados.append(subject_id)

    df_limpio = df_maestro.loc[indices_a_mantener]
    total_limpio = len(df_limpio)

    df_limpio.to_csv(archivo_maestro, index=False)

    print(f"Pacientes originales en el CSV: {total_original}")
    print(f"Pacientes eliminados (corruptos/sin imagen): {len(pacientes_eliminados)}")
    print(f"Pacientes válidos guardados: {total_limpio}")
    
    if pacientes_eliminados:
        print("\nIDs de los pacientes eliminados del entrenamiento:")
        print(pacientes_eliminados)

    print(f"\n[OK] El archivo {archivo_maestro} ha sido purgado con éxito.")
    print("Ejecuta de nuevo 'python scripts/validacion/validacion_datos_procesados.py'. Debería dar PASSED.")

if __name__ == "__main__":
    main()