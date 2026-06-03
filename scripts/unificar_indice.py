import pandas as pd
import os
import sys

# --- CONFIGURACIÓN DE RUTAS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dir_indexs = os.path.join(project_root, 'data', 'indexs')
archivo_salida = os.path.join(project_root, 'data_index.csv')

# Archivos base
csv_pd_control = os.path.join(dir_indexs, 'ppmi_pd_control.csv')
csv_prodromal = os.path.join(dir_indexs, 'ppmi_prodromal.csv')

def main():
    print("="*60)
    print(" CONSTRUCCIÓN DEL ÍNDICE MAESTRO (data_index.csv)")
    print("="*60 + "\n")

    # 1. Comprobar que los archivos existen
    if not os.path.exists(csv_pd_control) or not os.path.exists(csv_prodromal):
        print("[ERROR] Faltan archivos CSV en data/indexs/. Ejecuta el script de validación primero.")
        sys.exit(1)

    print("[1/2] Uniendo metadatos de LONI...")
    
    # Cargar los CSV
    df_pd = pd.read_csv(csv_pd_control)
    df_prod = pd.read_csv(csv_prodromal)

    # Unir ambos DataFrames
    df_completo = pd.concat([df_pd, df_prod], ignore_index=True)

    # 2. Filtrar y crear la estructura para el modelo
    print("[2/2] Dando formato a las rutas y etiquetas...")
    
    datos_limpios = []

    for index, row in df_completo.iterrows():
        # Extraer ID y Etiqueta limpiando espacios
        subject_id = str(row['Subject']).strip()
        label = str(row['Group']).strip()
        
        # Crear la ruta relativa esperada de la imagen procesada
        nombre_imagen = f"{subject_id}_slice.png" 
        ruta_relativa = os.path.join('data', 'PPMI_Procesado_2D', nombre_imagen)
        
        datos_limpios.append({
            'subject_id': subject_id,
            'image_path': ruta_relativa,
            'label': label
        })

    # Guardar el resultado final
    df_maestro = pd.DataFrame(datos_limpios)
    df_maestro.to_csv(archivo_salida, index=False)

    print(f"\n[OK] Índice generado con éxito: {archivo_salida}")
    print(f"Total de pacientes en el índice: {len(df_maestro)}")
    
    print("\nDistribución de clases:")
    distribucion = df_maestro['label'].value_counts()
    for clase, cantidad in distribucion.items():
        print(f"  - {clase}: {cantidad} pacientes")

if __name__ == "__main__":
    main()