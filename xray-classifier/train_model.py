import time
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.asyncio import tqdm
from hybrid_cnn_transformer import HybridCNNTransformer
from parameters import NUM_EPOCHS, LEARNING_RATE

def train_model(proj_path, data_bundle, device, model_name, dry_run=True):
    model = HybridCNNTransformer.from_pretrained(
        proj_path + 'pretrained/hybrid_oct.pth',
        device=device,
        num_classes=4)
    model.change_num_classes(2, device)

    optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=data_bundle.classes_info.weights.to(device))

    model, train_loss, train_acc, val_loss, val_acc = _run_train_loop(
        model,
        data_bundle.loaders.train,
        data_bundle.loaders.val,
        criterion,
        optimizer,
        num_epochs=NUM_EPOCHS,
        model_name=model_name,
        dry_run=dry_run
    )
    return model, train_loss, train_acc, val_loss, val_acc


def _run_train_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_name,
        num_epochs,
        dry_run=True,
):
    """
    Обучение модели с сохранением лучшей версии в PyTorch и ONNX форматах
    и комплексным логированием в MLflow.
    Возвращает лучшую модель по точности на валидационном наборе
    """

    best_accuracy = 0.0
    best_model_weights = None
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    total_start_time = time.time()
    device = next(model.parameters()).device

    # Устанавливаем базовые теги
    mlflow.set_tags({
        "task": "classification",
        "framework": "pytorch",
        "data": "xray_images",
        "device": str(device)
    })

    # Логируем гиперпараметры
    mlflow.log_params({
        "num_epochs": num_epochs,
        "model_name": model_name,
        "model_type": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": train_loader.batch_size
    })

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Фаза обучения
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        if not dry_run:
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | Training"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
        else:
            train_total = 0.001

        # Фаза валидации
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        if not dry_run:
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | Validation"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
        else:
            val_total = 0.001

        # Вычисление метрик
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        # Вывод метрик в консоль
        print(f"\nEpoch {epoch + 1}/{num_epochs} metrics:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")
        print(f"  Epoch Time: {time.time() - epoch_start_time:.2f} seconds\n")

        # Логируем метрики для эпохи
        mlflow.log_metrics({
            "train_loss": epoch_train_loss,
            "train_accuracy": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_accuracy": epoch_val_acc,
            "epoch_time": time.time() - epoch_start_time
        }, step=epoch + 1)

        # Сохранение лучшей модели
        if epoch_val_acc > best_accuracy:
            best_accuracy = epoch_val_acc
            best_model_weights = model.state_dict()
            print(f"New best model found with accuracy {best_accuracy:.2f}% - saving model")

    # Логируем итоговую метрику качества и время обучения
    total_time = time.time() - total_start_time
    mlflow.log_metrics({
        "best_accuracy": best_accuracy,
        "total_training_time": total_time
    })

    # Логируем архитектуру модели и метрики в виде файлов
    mlflow.log_text(str(model), "model_architecture.txt")
    mlflow.log_dict({
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }, "training_metrics.json")

    # Загружаем в model лучшую модель
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval()

    return model, train_losses, train_accuracies, val_losses, val_accuracies
