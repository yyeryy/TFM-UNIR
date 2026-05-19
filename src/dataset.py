import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class ParkinsonDataset(Dataset):
    def __init__(self, df_imagenes, ruta_imagenes, transformaciones=None):
        self.df = df_imagenes.reset_index(drop=True)
        self.ruta_imagenes = Path(ruta_imagenes)
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


def preparar_dataloaders(ruta_csv, ruta_imagenes, batch_size=32):

    ruta_imagenes = Path(ruta_imagenes)
    
    df_clinico = pd.read_csv(ruta_csv)
    df_clinico['Subject'] = df_clinico['Subject'].astype(str).str.strip()
    df_unico = df_clinico.drop_duplicates(subset=['Subject']).copy()
    
    df_unico['Etiqueta'] = df_unico['Group'].apply(lambda x: 1 if x == 'PD' else 0)
    
    archivos_npy = [f.name for f in ruta_imagenes.glob("*.npy")]
    df_archivos = pd.DataFrame({'Archivo': archivos_npy})
    df_archivos['Subject'] = df_archivos['Archivo'].apply(lambda x: x.split('_')[0])
    
    df_master = pd.merge(df_archivos, df_unico[['Subject', 'Etiqueta']], on='Subject', how='inner')
    
    sujetos_unicos = df_master[['Subject', 'Etiqueta']].drop_duplicates()
    
    train_subj, resto_subj = train_test_split(sujetos_unicos, test_size=0.30, stratify=sujetos_unicos['Etiqueta'], random_state=42)
    val_subj, test_subj = train_test_split(resto_subj, test_size=0.50, stratify=resto_subj['Etiqueta'], random_state=42)
    
    df_train = df_master[df_master['Subject'].isin(train_subj['Subject'])]
    df_val = df_master[df_master['Subject'].isin(val_subj['Subject'])]
    df_test = df_master[df_master['Subject'].isin(test_subj['Subject'])]
    
    train_dataset = ParkinsonDataset(df_train, ruta_imagenes)
    val_dataset = ParkinsonDataset(df_val, ruta_imagenes)
    test_dataset = ParkinsonDataset(df_test, ruta_imagenes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    pd_count = len(df_train[df_train['Etiqueta'] == 1])
    control_count = len(df_train[df_train['Etiqueta'] == 0])
    pesos_clase = torch.tensor([pd_count / control_count, 1.0]).float()
    
    return train_loader, val_loader, test_loader, pesos_clase

if __name__ == "__main__":
    try:
        ruta_csv_prueba = "data_index.csv"
        ruta_imagenes_prueba = "data/PPMI_Procesado_2D"
        train_l, val_l, test_l, pesos = preparar_dataloaders(ruta_csv_prueba, ruta_imagenes_prueba, batch_size=32)
        print("[OK] Modulo dataset compilado y cruce de datos exitoso. Listo para importar en main.py.")
    except Exception as e:
        print(f"[ERROR] Revisa las rutas o el CSV: {e}")