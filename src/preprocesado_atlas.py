import os
import ants
import numpy as np
import torchio as tio
from pathlib import Path
from tqdm import tqdm
from nilearn import datasets
import warnings
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

warnings.filterwarnings("ignore")
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

# DEFINICIÓN DE RUTAS
DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
RUTA_ORIGEN = DIRECTORIO_RAIZ / "data" / "PPMI"
RUTA_DESTINO = DIRECTORIO_RAIZ / "data" / "PPMI_Procesado_2D_Atlas"

NUMERO_DE_CORTES = 15

def seleccionar_mejor_secuencia(carpetas_dicom):
    """Selecciona la mejor secuencia y bloquea archivos basura."""
    # ESCUDO 1: Filtrar localizadores y calibraciones
    carpetas_validas = []
    for carpeta in carpetas_dicom:
        nombre = carpeta.upper()
        if "LOC" in nombre or "SCOUT" in nombre or "CALIBRATION" in nombre or "LOCALIZER" in nombre:
            continue
        carpetas_validas.append(carpeta)

    if not carpetas_validas:
        return None

    for carpeta in carpetas_validas:
        if "MPRAGE" in carpeta.upper() or "3D" in carpeta.upper(): return carpeta
    for carpeta in carpetas_validas:
        if "_RPT" in carpeta.upper(): return carpeta
        
    return carpetas_validas[0]

def procesar_un_sujeto(sujeto_path, ruta_t1_mni, ruta_mask_mni):
    try:
        id_sujeto = sujeto_path.name
        
        archivo_comprobacion = RUTA_DESTINO / f"{id_sujeto}_axial_00.npy"
        if archivo_comprobacion.exists():
            return "OMITIDO"
            
        carpetas_dicom = []
        for raiz, _, archivos in os.walk(sujeto_path):
            archivos_validos = [f for f in archivos if f.lower().endswith('.dcm') or f.lower().endswith('.ima')]
            if len(archivos_validos) > 30: carpetas_dicom.append(raiz)      

        if not carpetas_dicom: return "SIN_DICOM"
        
        secuencia_elegida = seleccionar_mejor_secuencia(carpetas_dicom)
        if not secuencia_elegida:
            return "SOLO_LOCALIZADORES"
            
        # Lectura inicial para auditoría
        img_tio = tio.ScalarImage(secuencia_elegida)
        
        # ESCUDO 2: Filtro Geométrico (Rechazar cortes gruesos)
        # Si el espaciado en X, Y o Z es mayor a 2.5mm, la imagen no es 3D de alta resolución
        espaciado_maximo = max(img_tio.spacing)
        if espaciado_maximo > 2.5:
            return f"DESCARTADO_ANISOTROPICO (Espaciado {espaciado_maximo:.1f}mm)"

        # 2. Carga y Orientación del Molde (MNI152)
        atlas_mni = ants.image_read(ruta_t1_mni)
        atlas_mni_ras = ants.reorient_image2(atlas_mni, 'RAS')
        
        mascara_mni = ants.image_read(ruta_mask_mni)
        mascara_mni_ras = ants.reorient_image2(mascara_mni, 'RAS')
        
        # 3. Lectura Robusta con TorchIO
        pipeline_previa = tio.Compose([
            tio.ToCanonical(),           
            tio.Resample((1.0, 1.0, 1.0)) 
        ])
        img_tio_pre = pipeline_previa(img_tio)

        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
            temp_path = tmp.name
        img_tio_pre.save(temp_path)

        # 4. Registro Rígido
        imagen_paciente_ras = ants.image_read(temp_path)
        registro = ants.registration(
            fixed=atlas_mni_ras, 
            moving=imagen_paciente_ras, 
            type_of_transform='Rigid'
        )
        volumen_alineado = registro['warpedmovout']
        os.remove(temp_path) 
        
        # 5. Skull Stripping
        volumen_sin_craneo = volumen_alineado * mascara_mni_ras
        
        # 6. Normalización Avanzada
        array_np = volumen_sin_craneo.numpy()
        tensor_final_tio = tio.ScalarImage(tensor=np.expand_dims(array_np, axis=0))
        
        pipeline_post = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
            tio.CropOrPad((224, 224, atlas_mni_ras.shape[2])) 
        ])
        imagen_procesada = pipeline_post(tensor_final_tio)
        matriz_final = imagen_procesada.data[0].numpy()
        
        # 7. Extracción de Cortes
        z_anatomico_parkinson = 75 
        corte_inicial = z_anatomico_parkinson - (NUMERO_DE_CORTES // 2)
        corte_final = corte_inicial + NUMERO_DE_CORTES
        
        contador_corte = 0
        for indice_z in range(corte_inicial, corte_final):
            corte_2d = matriz_final[:, :, indice_z]
            nombre_archivo = RUTA_DESTINO / f"{id_sujeto}_axial_{contador_corte:02d}.npy"
            np.save(nombre_archivo, corte_2d)
            contador_corte += 1
            
        return "OK"
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def procesar_dataset_con_atlas():
    RUTA_DESTINO.mkdir(parents=True, exist_ok=True)
    
    mni_data = datasets.fetch_icbm152_2009()
    ruta_t1_mni = mni_data['t1']
    ruta_mask_mni = mni_data['mask']
    
    carpetas_sujetos = [d for d in RUTA_ORIGEN.iterdir() if d.is_dir()]
    total_sujetos = len(carpetas_sujetos)
    
    nucleos_totales = multiprocessing.cpu_count()
    max_workers = 4

    print("="*60)
    print(" PIPELINE ROBUSTA: FILTRO QC + REGISTRO MNI152")
    print("="*60)

    procesados_ok = 0
    omitidos = 0
    errores = 0
    descartados_qc = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(procesar_un_sujeto, ruta_sujeto, ruta_t1_mni, ruta_mask_mni): ruta_sujeto 
            for ruta_sujeto in carpetas_sujetos
        }
        
        for futuro in tqdm(as_completed(futuros), total=total_sujetos, desc="Procesando"):
            resultado = futuro.result()
            if resultado == "OK": procesados_ok += 1
            elif resultado == "OMITIDO": omitidos += 1
            elif "DESCARTADO" in resultado or "SOLO_LOCALIZADORES" in resultado:
                descartados_qc += 1
            else:
                errores += 1

    print("\n" + "="*40)
    print(" RESUMEN FINAL DE PREPROCESAMIENTO")
    print("\n" + "="*40)
    print(f"Imágenes válidas y alineadas: {procesados_ok}")
    print(f"Omitidos (Ya existían): {omitidos}")
    print(f"Baja calidad bloqueados por QC: {descartados_qc}")
    print(f"Fallos de software: {errores}")
    print(f"Total tensores listas para entrenar: {(procesados_ok + omitidos) * NUMERO_DE_CORTES}")

if __name__ == "__main__":
    procesar_dataset_con_atlas()