import os
import random

import numpy as np
import torch

seed = 42


def seed_all():
    """
    Устанавливает seed для всех используемых библиотек для обеспечения воспроизводимости.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True  # Делает CuDNN детерминированным
    torch.backends.cudnn.benchmark = False  # Отключает авто-тюнинг (может менять поведение)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Для некоторых версий CUDA
    torch.use_deterministic_algorithms(True)


#
def create_torch_generator():
    """
    Создает и возвращает генератор случайных чисел PyTorch с фиксированным значением seed
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


#
def seed_worker(worker_id):
    """
    Функция для установки seed в DataLoader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
