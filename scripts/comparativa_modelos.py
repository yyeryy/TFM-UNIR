import sys
import os
from pathlib import Path

# Activar el fallback para Mac: Si MPS no soporta una operacion, que use la CPU sin romper el script
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

ruta_raiz = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ruta_raiz))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from src.dataset import preparar_dataloaders

def obtener_dispositivo():
    """Detecta de forma segura el mejor hardware disponible"""
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda:0")
        nombre_gpu = torch.cuda.get_device_name(0)
        print(f"[INFO] Hardware detectado: NVIDIA GPU ({nombre_gpu}) - Usando CUDA")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dispositivo = torch.device("mps")
        print("[INFO] Hardware detectado: Apple Silicon (M1/M2/M3) - Usando MPS")
    else:
        dispositivo = torch.device("cpu")
        print("[WARNING] No se detecto GPU compatible. Usando CPU (Sera muy lento)")
    return dispositivo

def configurar_modelo(tipo_modelo='resnet18', num_clases=2):
    """Configura ResNet18 o ResNet50 con Transfer Learning"""
    if tipo_modelo == 'resnet18':
        modelo = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = modelo.fc.in_features
        capas_a_descongelar = modelo.layer4
    elif tipo_modelo == 'resnet50':
        modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = modelo.fc.in_features
        capas_a_descongelar = modelo.layer4
    else:
        raise ValueError("Modelo no soportado")

    # Congelar toda la red
    for param in modelo.parameters():
        param.requires_grad = False
        
    # Descongelar el último bloque convolucional
    for param in capas_a_descongelar.parameters():
        param.requires_grad = True

    # Cambiar la capa final
    modelo.fc = nn.Linear(num_ftrs, num_clases)
    
    return modelo

def test_rapido(modelo, nombre, train_loader, val_loader, pesos_clase, device, num_epochs=10):
    print(f"\n{'='*60}")
    print(f" INICIANDO ENTRENAMIENTO: {nombre.upper()}")
    print(f"{'='*60}")
    
    modelo = modelo.to(device)
    pesos_clase = pesos_clase.to(device)
    criterio = nn.CrossEntropyLoss(weight=pesos_clase)
    
    parametros_a_entrenar = [p for p in modelo.parameters() if p.requires_grad]
    optimizador = optim.Adam(parametros_a_entrenar, lr=0.001)
    
    resultados = []
    inicio_total = time.time()

    for epoch in range(num_epochs):
        inicio_epoch = time.time()
        metricas = {'Epoch': epoch + 1}
        
        for fase in ['train', 'val']:
            if fase == 'train':
                modelo.train()
                loader = train_loader
            else:
                modelo.eval()
                loader = val_loader

            running_loss = 0.0
            
            # Listas para guardar las etiquetas reales y las predicciones de toda la época
            todas_preds = []
            todas_labels = []

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizador.zero_grad()

                with torch.set_grad_enabled(fase == 'train'):
                    outputs = modelo(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterio(outputs, labels)

                    if fase == 'train':
                        loss.backward()
                        optimizador.step()

                running_loss += loss.item() * inputs.size(0)
                
                # Guardar para el cálculo de métricas de Sklearn
                todas_preds.extend(preds.cpu().numpy())
                todas_labels.extend(labels.cpu().numpy())

            # Calcular métricas de la época
            epoch_loss = running_loss / len(todas_labels)
            
            epoch_prec = precision_score(todas_labels, todas_preds, average='binary', zero_division=0)
            epoch_rec = recall_score(todas_labels, todas_preds, average='binary', zero_division=0)
            epoch_f1 = f1_score(todas_labels, todas_preds, average='binary', zero_division=0)
            
            # Accuracy manual para evitar incompatibilidad de .double() en Mac MPS
            epoch_acc = sum([1 for p, l in zip(todas_preds, todas_labels) if p == l]) / len(todas_labels)

            metricas[f'{fase.capitalize()} Loss'] = round(epoch_loss, 4)
            metricas[f'{fase.capitalize()} Acc'] = round(epoch_acc, 4)
            metricas[f'{fase.capitalize()} Precision'] = round(epoch_prec, 4)
            metricas[f'{fase.capitalize()} Recall'] = round(epoch_rec, 4)
            metricas[f'{fase.capitalize()} F1'] = round(epoch_f1, 4)

        tiempo_epoch = time.time() - inicio_epoch
        metricas['Tiempo (s)'] = round(tiempo_epoch, 1)
        
        # Indicador de Overfitting
        metricas['Riesgo Overfitting'] = round(metricas['Train Acc'] - metricas['Val Acc'], 4)
        
        resultados.append(metricas)
        print(f"Ep {epoch+1:02d} | Train Loss: {metricas['Train Loss']:.4f} | Val Loss: {metricas['Val Loss']:.4f} | "
              f"Val Acc: {metricas['Val Acc']:.4f} | Val Recall: {metricas['Val Recall']:.4f} | Val F1: {metricas['Val F1']:.4f}")

    tiempo_total = time.time() - inicio_total
    df_resultados = pd.DataFrame(resultados)
    
    return df_resultados, tiempo_total

def generar_graficos(df_r18, df_r50, dir_raiz):
    """Genera un panel de 2x2 y lo guarda en la carpeta /graficas"""
    print("\n[INFO] Generando gráficos comparativos avanzados...")
    
    # 1. Crear carpeta graficas/ si no existe
    dir_graficas = Path(dir_raiz) / 'graficas'
    dir_graficas.mkdir(parents=True, exist_ok=True)
    ruta_guardado = dir_graficas / 'comparativa_avanzada_modelos.png'

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Gráfico de Loss (Train vs Val)
    axs[0, 0].plot(df_r18['Epoch'], df_r18['Train Loss'], 'b--', label='R18 Train Loss', alpha=0.6)
    axs[0, 0].plot(df_r18['Epoch'], df_r18['Val Loss'], 'b-', label='R18 Val Loss', linewidth=2)
    axs[0, 0].plot(df_r50['Epoch'], df_r50['Train Loss'], 'r--', label='R50 Train Loss', alpha=0.6)
    axs[0, 0].plot(df_r50['Epoch'], df_r50['Val Loss'], 'r-', label='R50 Val Loss', linewidth=2)
    axs[0, 0].set_title('Evolución de Pérdida (Overfitting Check)', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Épocas')
    axs[0, 0].set_ylabel('Loss (CrossEntropy)')
    axs[0, 0].legend()

    # 2. Gráfico de Accuracy
    axs[0, 1].plot(df_r18['Epoch'], df_r18['Val Acc'], 'b-o', label='ResNet-18', linewidth=2)
    axs[0, 1].plot(df_r50['Epoch'], df_r50['Val Acc'], 'r-s', label='ResNet-50', linewidth=2)
    axs[0, 1].set_title('Precisión Global (Validation Accuracy)', fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel('Épocas')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()

    # 3. Gráfico de Recall (Sensibilidad Médica)
    axs[1, 0].plot(df_r18['Epoch'], df_r18['Val Recall'], 'b-o', label='ResNet-18', linewidth=2)
    axs[1, 0].plot(df_r50['Epoch'], df_r50['Val Recall'], 'r-s', label='ResNet-50', linewidth=2)
    axs[1, 0].set_title('Sensibilidad (Validation Recall - Detección de PD)', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('Épocas')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].legend()

    # 4. Gráfico de F1-Score (Balance Prec-Rec)
    axs[1, 1].plot(df_r18['Epoch'], df_r18['Val F1'], 'b-o', label='ResNet-18', linewidth=2)
    axs[1, 1].plot(df_r50['Epoch'], df_r50['Val F1'], 'r-s', label='ResNet-50', linewidth=2)
    axs[1, 1].set_title('Métrica Combinada (Validation F1-Score)', fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel('Épocas')
    axs[1, 1].set_ylabel('F1-Score')
    axs[1, 1].legend()

    plt.suptitle('Evaluación de Arquitecturas: ResNet-18 vs ResNet-50 (PPMI Dataset)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    print(f"[OK] Gráfico guardado con éxito en:\n     -> {ruta_guardado}")

if __name__ == "__main__":
    device = obtener_dispositivo()
    
    ruta_csv = "data_index.csv"
    ruta_img = "data/PPMI_Procesado_2D"
    
    print("[INFO] Cargando Dataloaders...")
    train_loader, val_loader, test_loader, pesos, mapeo = preparar_dataloaders(
        ruta_csv, ruta_img, clases_permitidas=['Control', 'PD'], batch_size=32
    )
    
    EPOCHS_PRUEBA = 10
    
    # 1. Probar ResNet18
    modelo_r18 = configurar_modelo('resnet18', num_clases=2)
    df_r18, t_r18 = test_rapido(modelo_r18, "ResNet-18", train_loader, val_loader, pesos, device, num_epochs=EPOCHS_PRUEBA)
    
    # 2. Probar ResNet50
    modelo_r50 = configurar_modelo('resnet50', num_clases=2)
    df_r50, t_r50 = test_rapido(modelo_r50, "ResNet-50", train_loader, val_loader, pesos, device, num_epochs=EPOCHS_PRUEBA)
    
    # 3. Generar la imagen para el TFM en la carpeta de gráficas
    generar_graficos(df_r18, df_r50, ruta_raiz)
    
    # --- RESUMEN FINAL ---
    print("\n" + "="*60)
    print(" CONCLUSIÓN DE LA COMPARATIVA (ÉPOCA 10)")
    print("="*60)
    print(f"Tiempo Total ResNet-18: {t_r18/60:.2f} minutos")
    print(f"Tiempo Total ResNet-50: {t_r50/60:.2f} minutos")
    
    print("\n[Métricas Finales en Validación - ResNet-18]")
    print(f"Accuracy:  {df_r18.iloc[-1]['Val Acc']:.4f}")
    print(f"Precision: {df_r18.iloc[-1]['Val Precision']:.4f}")
    print(f"Recall:    {df_r18.iloc[-1]['Val Recall']:.4f}")
    print(f"F1-Score:  {df_r18.iloc[-1]['Val F1']:.4f}")
    
    print("\n[Métricas Finales en Validación - ResNet-50]")
    print(f"Accuracy:  {df_r50.iloc[-1]['Val Acc']:.4f}")
    print(f"Precision: {df_r50.iloc[-1]['Val Precision']:.4f}")
    print(f"Recall:    {df_r50.iloc[-1]['Val Recall']:.4f}")
    print(f"F1-Score:  {df_r50.iloc[-1]['Val F1']:.4f}")