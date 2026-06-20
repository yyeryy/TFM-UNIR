import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import copy
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import normalizar_roi_frac, preparar_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento controlado ResNet para PPMI")

    # MODIFICACIÓN CRÍTICA: Actualizamos las rutas por defecto a la nueva base de datos MNI152
    parser.add_argument("--csv", type=str, default="data_index.csv")
    parser.add_argument("--images", type=str, default="data/PPMI_Procesado_2D_Atlas")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--excel", type=str, default="resultados/experimentos_entrenamiento.xlsx")

    # Modelo y Clases
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    parser.add_argument("--classes", nargs="+", default=["Control", "PD"])
    parser.add_argument("--num-classes", type=int, default=2)

    # Hiperparámetros de entrenamiento
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Optimizador y Scheduler
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"])
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "none"])
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.3)

    # Early Stopping y Monitorización
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_f1",
        choices=["val_loss", "val_acc", "val_balanced_acc", "val_recall", "val_f1", "val_roc_auc"],
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    # ROI de entrada al modelo (recorte central)
    parser.add_argument(
        "--roi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activa el ROI de entrada (CenterCrop central reescalado a 224). Usa --no-roi para desactivarlo (Por defecto: activado)."
    )
    parser.add_argument(
        "--roi-frac",
        type=normalizar_roi_frac,
        default=0.6,
        help="Lado a conservar como fraccion o porcentaje (0.8 u 80 -> 179x179).",
    )

    # Configuración de red y hardware
    parser.add_argument("--freeze", type=str, default="layer4", choices=["head", "layer4", "none"])
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Naming y Logging
    parser.add_argument("--run-name", type=str, default=None, help="Forzar nombre. Si no, usará auto-versionado (v1, v2...)")
    parser.add_argument("--eval-test", action="store_true")
    
    # WandB activado por defecto
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activar o desactivar logging en Weights & Biases (Por defecto: activado)"
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no esta disponible.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            raise RuntimeError("MPS no esta disponible.")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        print(f"[INFO] GPU detectada: CUDA - {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("[INFO] GPU detectada: Apple Silicon MPS")
        return torch.device("mps")

    print("[WARNING] No se detecto GPU compatible. Usando CPU.")
    return torch.device("cpu")


def get_next_version(output_dir, model_name):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 1
        
    archivos_existentes = list(output_dir.glob(f"{model_name}_v*_best.pth"))
    versiones = []
    
    for archivo in archivos_existentes:
        try:
            partes = archivo.stem.split('_v')
            if len(partes) > 1:
                num_v = int(partes[1].split('_')[0])
                versiones.append(num_v)
        except (IndexError, ValueError):
            continue
            
    return max(versiones) + 1 if versiones else 1


def build_model(model_name, num_classes, pretrained, freeze):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    num_features = model.fc.in_features

    for param in model.parameters():
        param.requires_grad = False

    if freeze == "none":
        for param in model.parameters():
            param.requires_grad = True
    elif freeze == "layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif freeze == "head":
        pass

    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_optimizer(args, model):
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)


def get_scheduler(args, optimizer, mode):
    if args.scheduler == "none":
        return None
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.min_lr,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr,
    )


def safe_metric(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def compute_metrics(labels, preds, probs, loss_sum):
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    unique_labels = np.unique(labels)
    is_binary = len(unique_labels) == 2
    average = "binary" if is_binary else "macro"

    metrics = {
        "loss": loss_sum / len(labels),
        "acc": accuracy_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average=average, zero_division=0),
        "recall": recall_score(labels, preds, average=average, zero_division=0),
        "f1": f1_score(labels, preds, average=average, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
        "cohen_kappa": cohen_kappa_score(labels, preds),
        "num_samples": len(labels),
        "confusion_matrix": json.dumps(confusion_matrix(labels, preds).tolist()),
    }

    if is_binary and probs.ndim == 2 and probs.shape[1] == 2:
        positive_probs = probs[:, 1]
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        metrics.update({
            "roc_auc": safe_metric(lambda: roc_auc_score(labels, positive_probs)),
            "pr_auc": safe_metric(lambda: average_precision_score(labels, positive_probs)),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else None,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    else:
        metrics.update({
            "roc_auc": None, "pr_auc": None, "specificity": None,
            "sensitivity": metrics["recall"], "tn": None, "fp": None, "fn": None, "tp": None,
        })

    return metrics


def run_epoch(model, loader, criterion, optimizer, device, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    loss_sum = 0.0
    non_blocking = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if scaler is not None and scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        loss_sum += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    return compute_metrics(all_labels, all_preds, all_probs, loss_sum)


class EarlyStopping:
    def __init__(self, mode, patience, min_delta):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        if value is None:
            self.bad_epochs += 1
            return False, self.bad_epochs >= self.patience

        if self.best is None:
            self.best = value
            return True, False

        improved = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta

        if improved:
            self.best = value
            self.bad_epochs = 0
            return True, False

        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def model_state_to_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_results_excel(excel_path, summary_row, history_rows):
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    new_summary = pd.DataFrame([summary_row])
    new_history = pd.DataFrame(history_rows)

    if excel_path.exists():
        old_summary = pd.read_excel(excel_path, sheet_name="resumen")
        old_history = pd.read_excel(excel_path, sheet_name="historial_epocas")
        summary = pd.concat([old_summary, new_summary], ignore_index=True)
        history = pd.concat([old_history, new_history], ignore_index=True)
    else:
        summary = new_summary
        history = new_history

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        summary.to_excel(writer, sheet_name="resumen", index=False)
        history.to_excel(writer, sheet_name="historial_epocas", index=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- VERSIONADO AUTOMÁTICO ---
    if args.run_name:
        run_name = args.run_name
    else:
        version_num = get_next_version(output_dir, args.model)
        run_name = f"{args.model}_v{version_num}"

    # --- CONFIGURACIÓN WANDB ---
    if args.wandb:
        import wandb
        wandb.init(
            project="TFM-Parkinson-PPMI",
            entity="yyeryy-unir", 
            name=run_name,
            config=vars(args)
        )

    run_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    device = get_device(args.device)

    use_amp = args.amp and device.type == "cuda"
    if args.amp and device.type != "cuda":
        print("[INFO] AMP solo se activa en CUDA. En MPS/CPU se entrenara sin AMP.")

    checkpoint_path = output_dir / f"{run_name}_best.pth"
    excel_path = PROJECT_ROOT / args.excel

    train_loader, val_loader, test_loader, class_weights, class_map = preparar_dataloaders(
        ruta_csv=PROJECT_ROOT / args.csv,
        ruta_imagenes=PROJECT_ROOT / args.images,
        clases_permitidas=args.classes,
        batch_size=args.batch_size,
        roi=args.roi,
        roi_frac=args.roi_frac,
    )

    model = build_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        freeze=args.freeze,
    ).to(device)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(args, model)

    monitor_to_metric = {
        "val_loss": "loss", "val_acc": "acc", "val_balanced_acc": "balanced_acc",
        "val_recall": "recall", "val_f1": "f1", "val_roc_auc": "roc_auc",
    }
    monitor_metric = monitor_to_metric[args.monitor]
    mode = "min" if args.monitor == "val_loss" else "max"

    scheduler = get_scheduler(args, optimizer, mode)
    early_stopping = EarlyStopping(mode=mode, patience=args.patience, min_delta=args.min_delta)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("\n" + "=" * 80)
    print(f"Entrenamiento: {run_name}")
    print(f"Modelo: {args.model}")
    print(f"Dispositivo: {device}")
    print(f"Clases: {class_map}")
    print(f"Parametros entrenables: {trainable_params:,} / {total_params:,}")
    print(f"Monitor: {args.monitor}")
    print(f"Checkpoint mejor epoca: {checkpoint_path}")
    print(f"Excel resultados: {excel_path}")
    print("=" * 80 + "\n")

    history = []
    best_state = None
    best_epoch = None
    best_monitor_value = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model=model, loader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device, scaler=scaler,
        )
        val_metrics = run_epoch(
            model=model, loader=val_loader, criterion=criterion,
            optimizer=None, device=device,
        )

        monitor_value = val_metrics[monitor_metric]
        improved, should_stop = early_stopping.step(monitor_value)

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(monitor_value)
            else:
                scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        # Logging a WandB
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "learning_rate": lr_now,
                "Train/Loss": train_metrics["loss"],
                "Train/Acc": train_metrics["acc"],
                "Train/Recall": train_metrics["recall"],
                "Val/Loss": val_metrics["loss"],
                "Val/Acc": val_metrics["acc"],
                "Val/Recall": val_metrics["recall"],
                "Val/F1": val_metrics["f1"],
                "Val/ROC-AUC": val_metrics.get("roc_auc", 0)
            })

        row = {
            "run_name": run_name, "fecha_inicio": run_started_at, "epoch": epoch,
            "lr": lr_now, "monitor": args.monitor, "monitor_value": monitor_value, "is_best": improved,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        if improved:
            best_epoch = epoch
            best_monitor_value = monitor_value
            best_state = model_state_to_cpu(model)

            torch.save({
                "epoch": best_epoch,
                "model_name": args.model,
                "model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "class_map": class_map,
                "class_weights": class_weights.detach().cpu(),
                "args": vars(args),
                "best_monitor": args.monitor,
                "best_monitor_value": best_monitor_value,
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, checkpoint_path)

        print(
            f"Ep {epoch:03d}/{args.epochs} | LR {lr_now:.2e} | "
            f"Train Loss {train_metrics['loss']:.4f} | Val Loss {val_metrics['loss']:.4f} | "
            f"Val Acc {val_metrics['acc']:.4f} | Val Recall {val_metrics['recall']:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f} | "
            f"{'BEST' if improved else f'wait {early_stopping.bad_epochs}/{args.patience}'} | "
            f"{time.time() - epoch_start:.1f}s"
        )

        if should_stop:
            print(f"\n[EARLY STOPPING] Sin mejora durante {args.patience} epocas.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = None
    if args.eval_test:
        test_metrics = run_epoch(
            model=model, loader=test_loader, criterion=criterion,
            optimizer=None, device=device,
        )

        print("\n[TEST]")
        print(f"Loss {test_metrics['loss']:.4f} | Acc {test_metrics['acc']:.4f} | "
              f"Recall {test_metrics['recall']:.4f} | F1 {test_metrics['f1']:.4f} | ROC-AUC {test_metrics['roc_auc']}")
        
        if args.wandb:
            wandb.log({f"Test/{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))})

    elapsed_minutes = (time.time() - start_time) / 60
    run_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_history_row = next((r for r in history if r["epoch"] == best_epoch), None)

    summary_row = {
        "run_name": run_name, "fecha_inicio": run_started_at, "fecha_fin": run_finished_at,
        "duracion_min": elapsed_minutes, "model": args.model, "device": str(device),
        "checkpoint_path": str(checkpoint_path), "excel_path": str(excel_path),
        "best_epoch": best_epoch, "best_monitor": args.monitor,
        "best_monitor_value": best_monitor_value, "class_map": json.dumps(class_map),
        "train_batches": len(train_loader), "val_batches": len(val_loader), "test_batches": len(test_loader),
        "trainable_params": trainable_params, "total_params": total_params, **vars(args),
    }

    if best_history_row:
        summary_row.update({f"best_{k}": v for k, v in best_history_row.items()})
    if test_metrics:
        summary_row.update({f"test_{k}": v for k, v in test_metrics.items()})

    save_results_excel(excel_path, summary_row, history)
    
    if args.wandb:
        wandb.finish()

    print("\n" + "=" * 80)
    print("Entrenamiento terminado")
    print(f"Tiempo total: {elapsed_minutes:.2f} min")
    print(f"Mejor epoca: {best_epoch} (Guardada en: {checkpoint_path})")
    print("=" * 80)


if __name__ == "__main__":
    main()
