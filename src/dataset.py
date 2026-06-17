import os
import platform
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import ResNet50_Weights

class ParkinsonDataset(Dataset):
    def __init__(self, df_imagenes, ruta_imagenes, transformaciones=None):
        self.df = df_imagenes.reset_index(drop=True)
        self.ruta_imagenes = Path(ruta_imagenes)
        
        if transformaciones is None:
            transform_oficial = ResNet50_Weights.DEFAULT.transforms()

            # 2. Le extraemos sus atributos nativos de media y desviación estándar
            mean_oficial = transform_oficial.mean  # Extrae automáticamente [0.485, 0.456, 0.406]
            std_oficial = transform_oficial.std
            
            self.transformaciones = transforms.Compose([
                transforms.Normalize(mean=mean_oficial, std=std_oficial)
            ])
        else:
            self.transformaciones = transformaciones

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        nombre_archivo = self.df.loc[idx, 'Archivo']
        etiqueta = self.df.loc[idx, 'Etiqueta']
        
        ruta_completa = self.ruta_imagenes / nombre_archivo
        imagen_2d = np.load(ruta_completa)
        
        tensor_imagen = torch.from_numpy(imagen_2d).float()
        
        tensor_imagen = tensor_imagen.unsqueeze(0)
        tensor_imagen = tensor_imagen.repeat(3, 1, 1)
        
        if self.transformaciones:
            tensor_imagen = self.transformaciones(tensor_imagen)
            
        return tensor_imagen, torch.tensor(etiqueta, dtype=torch.long)


def preparar_dataloaders(ruta_csv, ruta_imagenes, clases_permitidas=['Control', 'PD'], batch_size=32):
    ruta_imagenes = Path(ruta_imagenes)
    
    # 1. Cargar y limpiar
    df_clinico = pd.read_csv(ruta_csv)
    
    # Fix para Mac/Windows si quedaran decimales o espacios en el CSV
    df_clinico['Subject'] = df_clinico['Subject'].astype(str).str.strip().apply(lambda x: x.split('.')[0])
    df_clinico['Group'] = df_clinico['Group'].astype(str).str.strip()
    
    # 2. FILTRO DINÁMICO: Quedarse solo con las clases solicitadas
    df_clinico = df_clinico[df_clinico['Group'].isin(clases_permitidas)]
    
    df_unico = df_clinico.drop_duplicates(subset=['Subject']).copy()
    
    # 3. Mapeo automático de etiquetas (Ej: Control=0, PD=1, Prodromal=2)
    mapeo_etiquetas = {clase: idx for idx, clase in enumerate(clases_permitidas)}
    df_unico['Etiqueta'] = df_unico['Group'].map(mapeo_etiquetas)
    
    # 4. Vincular con los archivos físicos
    archivos_npy = [f.name for f in ruta_imagenes.glob("*.npy")]
    df_archivos = pd.DataFrame({'Archivo': archivos_npy})
    df_archivos['Subject'] = df_archivos['Archivo'].apply(lambda x: x.split('_')[0])
    
    # Inner join para asegurar que solo cargamos archivos de los sujetos filtrados
    df_master = pd.merge(df_archivos, df_unico[['Subject', 'Etiqueta']], on='Subject', how='inner')
    
    sujetos_unicos = df_master[['Subject', 'Etiqueta']].drop_duplicates()
    
    # 5. Splits a nivel de paciente (Evita Data Leakage entre cortes del mismo paciente)
    train_subj, resto_subj = train_test_split(sujetos_unicos, test_size=0.30, stratify=sujetos_unicos['Etiqueta'], random_state=42)
    val_subj, test_subj = train_test_split(resto_subj, test_size=0.50, stratify=resto_subj['Etiqueta'], random_state=42)
    
    df_train = df_master[df_master['Subject'].isin(train_subj['Subject'])]
    df_val = df_master[df_master['Subject'].isin(val_subj['Subject'])]
    df_test = df_master[df_master['Subject'].isin(test_subj['Subject'])]
    
    # --- FIX MULTIPLATAFORMA PARA DATALOADERS ---
    sistema = platform.system()
    trabajadores = 0 if sistema == 'Darwin' else 4
    usar_pin_memory = False if sistema == 'Darwin' else True
    print(
        f"[INFO] SO Detectado: {sistema}. "
        f"DataLoader configurado con num_workers={trabajadores}, pin_memory={usar_pin_memory}."
    )

    # 6. Crear Datasets y Loaders
    train_dataset = ParkinsonDataset(df_train, ruta_imagenes)
    val_dataset = ParkinsonDataset(df_val, ruta_imagenes)
    test_dataset = ParkinsonDataset(df_test, ruta_imagenes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=trabajadores, pin_memory=usar_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=trabajadores, pin_memory=usar_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=trabajadores, pin_memory=usar_pin_memory)
    
    conteos_clase = df_train['Etiqueta'].value_counts().sort_index()
    total_muestras = len(df_train)
    pesos_clase = torch.tensor([total_muestras / (len(clases_permitidas) * c) for c in conteos_clase]).float()
    
    return train_loader, val_loader, test_loader, pesos_clase, mapeo_etiquetas


if __name__ == "__main__":
    print("\n" + "="*50)
    print(" TEST MULTIPLATAFORMA DEL DATALOADER")
    print("="*50)
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        # MODIFICACIÓN: Apuntamos al csv maestro final
        ruta_csv = project_root / "data_index.csv" 
        
        # MODIFICACIÓN CRÍTICA: Apuntar a la nueva carpeta de imágenes limpias del Atlas
        ruta_img = project_root / "data" / "PPMI_Procesado_2D_Atlas"
        
        if torch.cuda.is_available():
            disp = "NVIDIA GPU (CUDA)"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            disp = "Apple Silicon (MPS)"
        else:
            disp = "CPU"
        print(f"[INFO] Procesador gráfico detectado: {disp}\n")

        t_loader, v_loader, test_loader, pesos, mapeo = preparar_dataloaders(
            ruta_csv, ruta_img, clases_permitidas=['Control', 'PD'], batch_size=32
        )
        print(f"\n[OK] Dataloaders creados con éxito.")
        print(f"     Mapeo de clases: {mapeo}")
        print(f"     Pesos de balanceo: {pesos.tolist()}")
        
        batch_imagenes, batch_etiquetas = next(iter(t_loader))
        print(f"\n[OK] Extracción de lote de prueba superada.")
        print(f"     Tensor de Imágenes: {batch_imagenes.shape} (Lote, Canales, Alto, Ancho)")
        print(f"     Tensor de Etiquetas: {batch_etiquetas.shape}")
        
    except Exception as e:
        print(f"[ERROR FATAL]: {e}")