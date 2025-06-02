import mlflow
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset, WeightedRandomSampler
from torchvision import transforms, datasets

from parameters import IMAGE_SIZE, BATCH_SIZE
from seed_initializer import create_torch_generator, seed_worker


class DatasetSizes:
    def __init__(self, train: int, val: int, test: int):
        self.train: int = train
        self.val: int = val
        self.test: int = test


class ClassesInfo:
    def __init__(self, weights: torch.Tensor, names: list[str], counts: torch.Tensor):
        self.weights: torch.Tensor = weights
        self.names: list[str] = names
        self.counts: torch.Tensor = counts


class DatasetLoaders:
    def __init__(self, train: DataLoader, val: DataLoader, test: DataLoader):
        self.train: DataLoader = train
        self.val = val
        self.test = test


class DataBundle:
    def __init__(
            self,
            dataset_loaders: DatasetLoaders,
            dataset_sizes: DatasetSizes,
            classes_info: ClassesInfo,
            image_size: tuple[int, int],
    ):
        self.loaders: DatasetLoaders = dataset_loaders
        self.dataset_sizes: DatasetSizes = dataset_sizes
        self.classes_info: ClassesInfo = classes_info
        self.image_size: tuple[int, int] = image_size


def _create_data_bundle(
        train_data_root_folder,
        test_data_root_folder,
        generator,
        seed_worker_fn,
        batch_size,
        num_workers,
        image_size
) -> DataBundle:
    """
    Создает и возвращает набор DataLoader'ов для обучения, валидации и тестирования.
    Возвращает:
        dict: Словарь с DataLoader'ами и дополнительной информацией:
            - 'loaders': словарь с train, val и test DataLoader'ами
            - 'class_weights': веса классов
            - 'class_names': имена классов
            - 'dataset_sizes': размеры наборов данных
    """

    # Базовые преобразования
    base_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # Загрузка данных для получения информации о классах
    full_train_data = datasets.ImageFolder(root=train_data_root_folder, transform=base_transform)
    test_data = datasets.ImageFolder(root=test_data_root_folder, transform=base_transform)

    # Сохраняем имена классов
    class_names = full_train_data.classes

    # Разделение на train и validation
    val_size = int(0.2 * len(full_train_data))
    train_size = len(full_train_data) - val_size
    train_data_base, val_data_base = random_split(full_train_data, [train_size, val_size], generator=generator)

    # Преобразования с аугментациями
    train_transform_aug = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
    ])

     # Создаем аугментированную версию ТОЛЬКО тренировочных данных
    train_data_aug = Subset(
        datasets.ImageFolder(root=train_data_root_folder, transform=train_transform_aug),
        indices=train_data_base.indices  # те же индексы, что и у train_data
    )

    # Комбинированные наборы данных
    combined_train_data = ConcatDataset([train_data_base, train_data_aug])
    combined_val_data = val_data_base  # Используем только базовые данные для валидации

    # Функция для получения меток
    def get_labels(dataset):
        if isinstance(dataset, Subset):
            return torch.tensor([dataset.dataset.targets[i] for i in dataset.indices])
        elif isinstance(dataset, ConcatDataset):
            return torch.cat([get_labels(sub) for sub in dataset.datasets])
        elif hasattr(dataset, 'targets'):
            return torch.tensor(dataset.targets)
        return torch.tensor([label for _, label in dataset])

    # Вычисляем веса классов
    train_labels = get_labels(combined_train_data)
    class_counts = torch.bincount(train_labels)
    class_weights = 1. / class_counts.float()
    class_weights = class_weights / class_weights.sum()

    # Создание DataLoader'ов
    def create_loader(dataset, shuffle=False, sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=seed_worker_fn,
            generator=generator,
        )

    # Сэмплер для балансировки классов
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    # Создаем DataLoader'ы
    train_loader = create_loader(combined_train_data, sampler=sampler)
    val_loader = create_loader(combined_val_data)
    test_loader = create_loader(test_data)

    return DataBundle(
        dataset_loaders=DatasetLoaders(
            train=train_loader,
            val=val_loader,
            test=test_loader
        ),
        classes_info=ClassesInfo(
            weights=class_weights,
            names=class_names,
            counts=class_counts
        ),
        dataset_sizes=DatasetSizes(
            train=len(combined_train_data),
            val=len(combined_val_data),
            test=len(test_data)
        ),
        image_size=image_size
    )


def _log_data_bundle(data_bundle: DataBundle):
    """
    Логирует информацию о загруженных данных.
    """
    output = []
    output.append("Dataset Sizes:")
    output.append(f"  Train: {data_bundle.dataset_sizes.train}")
    output.append(f"  Val: {data_bundle.dataset_sizes.val}")
    output.append(f"  Test: {data_bundle.dataset_sizes.test}")
    output.append("")
    output.append("Class Information:")
    for name, weight, count in zip(data_bundle.classes_info.names, data_bundle.classes_info.weights,
                                   data_bundle.classes_info.counts):
        output.append(f"  Class: {name}, Weight: {weight:.4f}, Count: {count.item()}")
    output.append("")
    output.append("Batch dimension example:")
    batch = next(iter(data_bundle.loaders.train))
    output.append(f"  {batch[0].shape}")
    text = '\n'.join(output)
    mlflow.log_text(text, "data_bundle_info.txt")
    print(text)


def get_data_bundle(project_path: str, num_workers=0) -> DataBundle:
    data_bundle = _create_data_bundle(
        train_data_root_folder=project_path + "datasets/chest_xray_clean/train",
        test_data_root_folder=project_path + "datasets/chest_xray_clean/test",
        generator=create_torch_generator(),
        seed_worker_fn=seed_worker,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )
    _log_data_bundle(data_bundle)
    return data_bundle
