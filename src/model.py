"""
model.py - Определяет архитектуру модели.

Мы используем transfer learning (перенос обучения):
- Берем предобученную на ImageNet модель (ResNet18)
- Заменяем последний слой на свой (2 класса)
- Обучаем только новый слой (или всю модель с маленьким lr)
"""

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
        model_name: пока поддерживается только 'resnet18'
        num_classes: количество классов (2 для cats/dogs)
        freeze_features: замораживать ли веса предобученной части

    Возвращает:
        model: готовая модель
    """

    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        # fc (fully connected) — последний слой ResNet: 512 признаков после
        # global average pooling -> 1000 классов ImageNet.
        # Заменяем его на свой: те же 512 входов -> num_classes выходов.
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Модель {model_name} не поддерживается")

    # Замораживаем веса предобученной части
    if freeze_features:
        # requires_grad=False: не считать градиент -> веса не обновляются
        for param in model.parameters():
            param.requires_grad = False
        # Размораживаем только новый классификатор (веса и смещение fc)
        for param in model.fc.parameters():
            param.requires_grad = True

    return model
