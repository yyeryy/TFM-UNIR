import pandas as pd
import os
import sys

# Configuracion de rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dir_indexs = os.path.join(project_root, 'data', 'indexs')
archivo_salida = os.path.join(project_root, 'data_index.csv')

csv_pd_control = os.path.join(dir_indexs, 'ppmi_pd_control.csv')
csv_prodromal = os.path.join(dir_indexs, 'ppmi_prodromal.csv')

def main():
    print("="*60)
    print(" CONSTRUCCIÓN DEL ÍNDICE MAESTRO (data_index.csv)")
    print("="*60 + "\n")

    if not os.path.exists(csv_pd_control) or not os.path.exists(csv_prodromal):
        print("[ERROR] Faltan archivos CSV en data/indexs/.")
        sys.exit(1)

    print("[1/2] Uniendo metadatos originales de LONI...")
    df_pd = pd.read_csv(csv_pd_control)
    df_prod = pd.read_csv(csv_prodromal)

    # Unir sin alterar columnas (mantiene la estructura original exacta)
    df_completo = pd.concat([df_pd, df_prod], ignore_index=True)

    print("[2/2] Guardando data_index.csv con la estructura original...")
    df_completo.to_csv(archivo_salida, index=False)

    print(f"\n[OK] Índice generado: {archivo_salida}")
    print(f"Total de registros: {len(df_completo)}")
    
    print("\nDistribución de clases ('Group'):")
    if 'Group' in df_completo.columns:
        for clase, cantidad in df_completo['Group'].value_counts().items():
            print(f"  - {clase}: {cantidad} pacientes")

if __name__ == "__main__":
    main()