import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

CSV_PATH = Path("data_index.csv")
DATA_DIR = Path("data/PPMI")
METADATA_DIR = Path("metadata")

TOTAL_SUJETOS_ESPERADOS = 954
TOTAL_XML_ESPERADOS = 823

def comprobar_dataset():
    rutas_correctas = True
    for ruta, nombre in [(CSV_PATH, "Archivo data_index.csv"), 
                         (DATA_DIR, "Carpeta de imágenes (data/PPMI)"), 
                         (METADATA_DIR, "Carpeta de metadatos (metadata)")]:
        if not ruta.exists():
            print(f"[ERROR] No se encuentra: {nombre} en '{ruta}'")
            rutas_correctas = False
            
    if not rutas_correctas:
        print("\nESTADO GLOBAL: KO (Faltan componentes principales en el directorio)")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        sujetos_csv = set(df['Subject'].dropna().astype(str).str.strip())
    except Exception as e:
        print(f"[ERROR] Fallo al leer el archivo CSV Maestro: {e}")
        print("\nESTADO GLOBAL: KO")
        return

    sujetos_carpetas = set(os.listdir(DATA_DIR))

    sujetos_xml = set()
    for xml_path in METADATA_DIR.glob("*.xml"):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            sub_id_elem = root.find(".//subjectIdentifier")
            if sub_id_elem is not None and sub_id_elem.text:
                sujetos_xml.add(sub_id_elem.text.strip())
        except Exception:
            pass

    datos_validos = True

    if sujetos_csv == sujetos_carpetas and len(sujetos_carpetas) == TOTAL_SUJETOS_ESPERADOS:
        print(f"[OK] Correspondencia exacta entre CSV y carpetas ({len(sujetos_carpetas)} sujetos).")
    else:
        print("[ERROR] Descuadre entre los sujetos del CSV y las carpetas descargadas en disco.")
        datos_validos = False

    if len(sujetos_xml) == TOTAL_XML_ESPERADOS and sujetos_xml.issubset(sujetos_carpetas):
        print(f"[OK] Estructura de metadatos XML integrada correctamente ({len(sujetos_xml)} IDs únicos).")
    else:
        print("[ERROR] El recuento o la identidad de los archivos XML no coincide con el estándar.")
        datos_validos = False

    if datos_validos:
        print("ESTADO GLOBAL: DATASET OK (Listo para trabajar)")
    else:
        print("ESTADO GLOBAL: DATASET KO (Revisar errores)")

if __name__ == "__main__":
    comprobar_dataset()