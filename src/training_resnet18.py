from pathlib import Path
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import preparar_dataloaders

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels).item()
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc


def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    RUTA_CSV = PROJECT_ROOT / "data_index.csv"
    RUTA_IMAGENES = PROJECT_ROOT / "data" / "PPMI_Procesado_2D"
    RUTA_MODELOS = PROJECT_ROOT / "models"
    RUTA_MODELOS.mkdir(exist_ok=True)

    RUTA_CHECKPOINT = RUTA_MODELOS / "resnet18_ppmi_best.pth"

    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2
    RANDOM_SEED = 42

    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Usando dispositivo: {device}")

    print("\nCargando datos...")

    train_loader, val_loader, test_loader, pesos_clase = preparar_dataloaders(
        ruta_csv=RUTA_CSV,
        ruta_imagenes=RUTA_IMAGENES,
        batch_size=BATCH_SIZE
    )

    pesos_clase = pesos_clase.to(device)

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Pesos de clase: {pesos_clase}")

    print("\nCreando modelo ResNet18...")

    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("ResNet18 cargado con pesos preentrenados de ImageNet.")
    except Exception as e:
        print("No se pudieron cargar los pesos preentrenados.")
        print(f"Motivo: {e}")
        print("Se usará ResNet18 sin pesos preentrenados.")
        model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=pesos_clase)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nComenzando entrenamiento...\n")

    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Época {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "history": history,
                    "model_name": "resnet18",
                    "num_classes": NUM_CLASSES,
                    "ruta_csv": str(RUTA_CSV),
                    "ruta_imagenes": str(RUTA_IMAGENES),
                },
                RUTA_CHECKPOINT
            )
            print(f"Nuevo mejor modelo guardado en: {RUTA_CHECKPOINT}")

        print()

    elapsed_time = time.time() - start_time

    print("Entrenamiento terminado")
    print(f"Tiempo total: {elapsed_time / 60:.2f} minutos")
    print(f"Mejor accuracy de validación: {best_val_acc:.4f}")
    print(f"Checkpoint guardado en: {RUTA_CHECKPOINT}")

    model.load_state_dict(best_model_state)
    model.eval()
    print("\nModelo final cargado con los mejores pesos de validación.")

if __name__ == "__main__":
    main()