import os
import time

import mlflow
import torch
import torchvision
from mlflow.models import infer_signature
from tqdm.asyncio import tqdm


def train_model_mlflow(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, model_name="best_model"):
    """
    Обучение модели с сохранением лучшей версии в PyTorch и ONNX форматах
    и комплексным логированием в MLflow.
    Возвращает лучшую модель по точности на валидационном наборе
    """
    # Создаем папки для сохранения моделей
    # os.makedirs("models", exist_ok=True)
    # os.makedirs("onnx_models", exist_ok=True)
    # os.makedirs("plots", exist_ok=True)

    best_accuracy = 0.0
    best_model_weights = None
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    total_start_time = time.time()
    device = next(model.parameters()).device  # Получаем устройство модели

    # Определяем требования для воспроизводимости
    pip_requirements = [
        f"torch=={torch.__version__}",
        f"torchvision=={torchvision.__version__}",
        "mlflow>=2.0"
    ]

    # Начинаем эксперимент MLflow

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

    # Подготовка примера данных для сигнатуры модели
    sample_batch = next(iter(train_loader))
    sample_input = sample_batch[0][0].unsqueeze(0).to(device)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Фаза обучения
        model.train()
        train_loss = 0.0
        train_correct = 0
        # train_total = 0
        train_total = 0.001

    #
    #     for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} | Training"):
    #         inputs, labels = inputs.to(device), labels.to(device)
    #
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         train_loss += loss.item()
    #         _, preds = torch.max(outputs, 1)
    #         train_correct += (preds == labels).sum().item()
    #         train_total += labels.size(0)
    #
        # Фаза валидации
        model.eval()
        val_loss = 0.0
        val_correct = 0
        # val_total = 0
        val_total = 0.001
    #
    #     with torch.no_grad():
    #         for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} | Validation"):
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
    #
    #             val_loss += loss.item()
    #             _, preds = torch.max(outputs, 1)
    #             val_correct += (preds == labels).sum().item()
    #             val_total += labels.size(0)
    #
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
        print(f"\nEpoch {epoch+1}/{num_epochs} metrics:")
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
        }, step=epoch+1)
    #
    #     # Сохранение лучшей модели
    #     if epoch_val_acc > best_accuracy:
    #         best_accuracy = epoch_val_acc
    #         best_model_weights = model.state_dict()
    #
    #         # 1. Сохраняем PyTorch модель
    #         torch_path = f"models/{model_name}.pth"
    #         torch.save({
    #             'state_dict': best_model_weights,
    #             'num_classes': 2,
    #             'model_type': model.__class__.__name__,
    #             'optimizer_state': optimizer.state_dict(),
    #             'epoch': epoch,
    #             'accuracy': best_accuracy
    #         }, torch_path)
    #
    #         print(f"New best model found with accuracy {best_accuracy:.2f}% - saving model")
    #         #mlflow.log_artifact(torch_path, "model_artifacts")
    #
    #         # Логируем модель в MLflow с сигнатурой и требованиями
    #         signature = infer_signature(
    #             sample_input.cpu().numpy(),
    #             model(sample_input).detach().cpu().numpy()
    #         )
    #         '''
    #         mlflow.pytorch.log_model(
    #             pytorch_model=model,
    #             artifact_path="pytorch_model",
    #             registered_model_name=model_name,
    #             signature=signature,
    #             input_example=sample_input.cpu().numpy(),
    #             pip_requirements=pip_requirements
    #         )'''
    #         print('end epoch')

    # Логируем модель в MLflow с сигнатурой и требованиями
    signature = infer_signature(
        sample_input.cpu().numpy(),
        model(sample_input).detach().cpu().numpy()
    )
    print('Сигнатура модели:', signature)

    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="pytorch_model",
        registered_model_name=model_name,
        signature=signature,
        input_example=sample_input.cpu().numpy(),
        pip_requirements=pip_requirements
    )



    # Финальное логирование
    total_time = time.time() - total_start_time
    mlflow.log_metrics({
        "best_accuracy": best_accuracy,
        "total_training_time": total_time
    })

    # Логируем финальные артефакты
    mlflow.log_text(str(model), "model_architecture.txt")
    mlflow.log_dict({
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }, "training_metrics.json")


    # Возвращаем лучшую модель
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval()

    return model, train_losses, train_accuracies, val_losses, val_accuracies
