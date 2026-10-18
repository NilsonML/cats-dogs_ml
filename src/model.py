"""
model.py - Определяет архитектуру модели.

Мы используем transfer learning (перенос обучения):
- Берем предобученную на ImageNet модель (ResNet18)
- Заменяем последний слой на свой (2 класса)
- Обучаем только новый слой (или всю модель с маленьким lr)
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(
    model_name: str = "resnet18",
    num_classes: int = 2,
    freeze_features: bool = True,
):
    """
    Создает модель для классификации.

    Аргументы:
        model_name: 'resnet18', 'resnet50', 'efficientnet_b0', etc.
        num_classes: количество классов (2 для cats/dogs)
        freeze_features: замораживать ли веса предобученной части

    Возвращает:
        model: готовая модель
    """

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = (
            model.fc.in_features
        )  # Здесь мы запоминаем число входных признаков у старого fc-слоя
        model.fc = nn.Linear(in_features, num_classes)  # Здесь мы заменяем старый fc-слой на новый.
        """
            fc — это сокращение от Fully Connected (полносвязный слой)

            in_features=x — размер вектора признаков, который приходит
            с предыдущего слоя (после global average pooling)

            out_features=x количество классов в датасете ImageNet
        """
    else:
        raise ValueError(f"Модель {model_name} не поддерживается")

    # Замораживаем веса предобученной части
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = (
                False  # говорит PyTorch: не вычислять градиент для этого параметра
            )
        # Размораживаем только новый классификатор
        for param in (
            model.fc.parameters() if hasattr(model, "fc") else model.classifier.parameters()
        ):
            param.requires_grad = True  # отдаёт веса и смещение только последнего слоя (нового классификатора, который мы создали ранее)

    return model


def load_checkpoint(model, checkpoint_path, device):
    """Загружает веса модели из файла."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model
