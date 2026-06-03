import os
import torchio as tio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

RUTA_ORIGEN = Path("data/PPMI")
RUTA_DESTINO = Path("data/PPMI_Procesado_2D")

ESPACIADO_OBJETIVO = (1.0, 1.0, 1.0)
TAMANO_MATRIZ_OBJETIVO = (224, 224, 192)
NUMERO_DE_CORTES = 15

def seleccionar_mejor_secuencia(carpetas_dicom):
    if len(carpetas_dicom) == 1:
        return carpetas_dicom[0]
        
    for carpeta in carpetas_dicom:
        if "_RPT" in carpeta.upper():
            return carpeta
            
    for carpeta in carpetas_dicom:
        if "MPRAGE" in carpeta.upper() or "3D" in carpeta.upper():
            return carpeta
            
    return carpetas_dicom[0]

def crear_pipeline():
    transformaciones = [
        tio.ToCanonical(),
        tio.Resample(ESPACIADO_OBJETIVO),
        tio.CropOrPad(TAMANO_MATRIZ_OBJETIVO),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    ]
    return tio.Compose(transformaciones)

def procesar_dataset_2d():
    RUTA_DESTINO.mkdir(parents=True, exist_ok=True)
    
    pipeline = crear_pipeline()
    carpetas_sujetos = [d for d in RUTA_ORIGEN.iterdir() if d.is_dir()]
    total_sujetos = len(carpetas_sujetos)
    
    procesados_ok = 0
    errores = 0

    for indice, sujeto_path in enumerate(tqdm(carpetas_sujetos, desc="Procesando pacientes", unit="paciente")):
        id_sujeto = sujeto_path.name
        actual = indice + 1
        
        archivo_comprobacion = RUTA_DESTINO / f"{id_sujeto}_axial_00.npy"
        if archivo_comprobacion.exists():
            print(f"Procesado {actual}/{total_sujetos} - Sujeto {id_sujeto} ya procesado. Omitiendo.")
            procesados_ok += 1
            continue
            
        carpetas_dicom = []
        for raiz, directorios, archivos in os.walk(sujeto_path):
            archivos_validos = [f for f in archivos if f.lower().endswith('.dcm') or f.lower().endswith('.ima')]
            
            if len(archivos_validos) > 30:
                carpetas_dicom.append(raiz)      

        if not carpetas_dicom:
            print(f"Procesado {actual}/{total_sujetos} - Sujeto {id_sujeto}: No se encontraron archivos DICOM.")
            errores += 1
            continue
            
        secuencia_elegida = seleccionar_mejor_secuencia(carpetas_dicom)
        
        try:
            imagen_bruta = tio.ScalarImage(secuencia_elegida)
            imagen_limpia = pipeline(imagen_bruta)
            
            tensor_3d = imagen_limpia.data[0]
            
            eje_z_total = tensor_3d.shape[2]
            centro_z = eje_z_total // 2
            
            corte_inicial = centro_z - (NUMERO_DE_CORTES // 2)
            corte_final = corte_inicial + NUMERO_DE_CORTES
            
            contador_corte = 0
            for indice_z in range(corte_inicial, corte_final):
                corte_2d = tensor_3d[:, :, indice_z].numpy()
                
                nombre_archivo = RUTA_DESTINO / f"{id_sujeto}_axial_{contador_corte:02d}.npy"
                np.save(nombre_archivo, corte_2d)
                contador_corte += 1
                
            procesados_ok += 1
            print(f"Procesado {actual}/{total_sujetos} - Sujeto {id_sujeto} procesado correctamente.")
                
        except Exception as e:
            print(f"Procesado {actual}/{total_sujetos} - Fallo en el sujeto {id_sujeto}: {str(e)}")
            errores += 1

    print("\n")
    print(f"Pacientes procesados correctamente: {procesados_ok}")
    print(f"Total de imagenes 2D generadas: {procesados_ok * NUMERO_DE_CORTES}")
    print(f"Pacientes con errores (descartados): {errores}")

if __name__ == "__main__":
    procesar_dataset_2d()