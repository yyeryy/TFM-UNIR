import os
import platform
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import ResNet50_Weights


def normalizar_roi_frac(valor):
    roi_frac = float(valor)
    if 1 < roi_frac <= 100:
        roi_frac /= 100.0
    if not 0 < roi_frac <= 1:
        raise ValueError("roi_frac debe estar entre 0 y 1, o expresarse como porcentaje entre 1 y 100.")
    return roi_frac

class ParkinsonDataset(Dataset):
    def __init__(self, df_imagenes, ruta_imagenes, transformaciones=None, return_subject=False):
        self.df = df_imagenes.reset_index(drop=True)
        self.ruta_imagenes = Path(ruta_imagenes)
        self.return_subject = return_subject
        
        if transformaciones is None:
            transform_oficial = ResNet50_Weights.DEFAULT.transforms()

            mean_oficial = transform_oficial.mean
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
            
        etiqueta_tensor = torch.tensor(etiqueta, dtype=torch.long)
        if self.return_subject:
            return tensor_imagen, etiqueta_tensor, str(self.df.loc[idx, 'Subject'])
        return tensor_imagen, etiqueta_tensor


def construir_transformaciones(roi=True, roi_frac=0.6):
    roi_frac = normalizar_roi_frac(roi_frac)
    transform_oficial = ResNet50_Weights.DEFAULT.transforms()
    mean_oficial = transform_oficial.mean  # [0.485, 0.456, 0.406]
    std_oficial = transform_oficial.std

    roi_prefix = []
    if roi:
        crop_size = int(round(224 * roi_frac))
        roi_prefix = [
            transforms.CenterCrop(crop_size),
            transforms.Resize((224, 224), antialias=True),
        ]
        print(f"[INFO] ROI de entrada ACTIVADO: CenterCrop({crop_size}) + Resize(224) (roi_frac={roi_frac}).")
    else:
        print("[INFO] ROI de entrada DESACTIVADO.")

    normalize = transforms.Normalize(mean=mean_oficial, std=std_oficial)

    train_transforms = transforms.Compose(
        roi_prefix + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            normalize,
        ]
    )
    eval_transforms = transforms.Compose(roi_prefix + [normalize])

    return train_transforms, eval_transforms


def preparar_dataloaders(
    ruta_csv,
    ruta_imagenes,
    clases_permitidas=['Control', 'PD'],
    batch_size=32,
    roi=True,
    roi_frac=0.6,
    balance_strategy="class_weights",
    return_subject=False,
):
    ruta_imagenes = Path(ruta_imagenes)
    
    df_clinico = pd.read_csv(ruta_csv)
    
    df_clinico['Subject'] = df_clinico['Subject'].astype(str).str.strip().apply(lambda x: x.split('.')[0])
    df_clinico['Group'] = df_clinico['Group'].astype(str).str.strip()
    
    df_clinico = df_clinico[df_clinico['Group'].isin(clases_permitidas)]
    
    df_unico = df_clinico.drop_duplicates(subset=['Subject']).copy()
    
    mapeo_etiquetas = {clase: idx for idx, clase in enumerate(clases_permitidas)}
    df_unico['Etiqueta'] = df_unico['Group'].map(mapeo_etiquetas)
    
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
    
    sistema = platform.system()
    trabajadores = 0 if sistema == 'Darwin' else 4
    usar_pin_memory = False if sistema == 'Darwin' else True
    print(
        f"[INFO] SO Detectado: {sistema}. "
        f"DataLoader configurado con num_workers={trabajadores}, pin_memory={usar_pin_memory}."
    )

    train_transforms, eval_transforms = construir_transformaciones(roi=roi, roi_frac=roi_frac)

    train_dataset = ParkinsonDataset(
        df_train,
        ruta_imagenes,
        transformaciones=train_transforms,
        return_subject=return_subject,
    )
    val_dataset = ParkinsonDataset(
        df_val,
        ruta_imagenes,
        transformaciones=eval_transforms,
        return_subject=return_subject,
    )
    test_dataset = ParkinsonDataset(
        df_test,
        ruta_imagenes,
        transformaciones=eval_transforms,
        return_subject=return_subject,
    )

    estrategias_validas = {"class_weights", "sampler", "none"}
    if balance_strategy not in estrategias_validas:
        raise ValueError(
            f"balance_strategy debe ser una de {sorted(estrategias_validas)}; "
            f"recibido: {balance_strategy}"
        )

    conteos_clase = df_train['Etiqueta'].value_counts().sort_index()
    conteos_clase = conteos_clase.reindex(range(len(clases_permitidas)), fill_value=0)
    if (conteos_clase == 0).any():
        raise ValueError("El split de entrenamiento no contiene muestras de todas las clases.")

    train_sampler = None
    if balance_strategy == "sampler":
        pesos_por_clase = (1.0 / conteos_clase).to_dict()
        pesos_muestras = df_train['Etiqueta'].map(pesos_por_clase).to_numpy(copy=True)
        generator = torch.Generator().manual_seed(42)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(pesos_muestras, dtype=torch.double),
            num_samples=len(pesos_muestras),
            replacement=True,
            generator=generator,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=trabajadores,
        pin_memory=usar_pin_memory,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=trabajadores, pin_memory=usar_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=trabajadores, pin_memory=usar_pin_memory)

    total_muestras = len(df_train)
    if balance_strategy == "class_weights":
        pesos_clase = torch.tensor([
            total_muestras / (len(clases_permitidas) * c)
            for c in conteos_clase
        ]).float()
    else:
        pesos_clase = torch.ones(len(clases_permitidas), dtype=torch.float32)

    print(
        f"[INFO] Balanceo: {balance_strategy}. "
        f"Train={len(df_train)} imagenes/{len(train_subj)} pacientes, "
        f"Val={len(df_val)}/{len(val_subj)}, Test={len(df_test)}/{len(test_subj)}."
    )
    
    return train_loader, val_loader, test_loader, pesos_clase, mapeo_etiquetas


if __name__ == "__main__":
    print("\n" + "="*50)
    print(" TEST MULTIPLATAFORMA DEL DATALOADER")
    print("="*50)
    try:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        ruta_csv = project_root / "data_index.csv" 
        
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
