"""
engine.py - Функции для обучения и оценки модели.

Здесь реализованы два ключевых цикла:
- train_one_epoch: обучение на одной эпохе
- evaluate: оценка на валидации
"""

import torch
from tqdm import tqdm
from typing import Dict, List, Tuple

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Обучает модель на одной эпохе.
    
    Аргументы:
        model: модель
        dataloader: загрузчик данных
        loss_fn: функция потерь
        optimizer: оптимизатор
        device: устройство (CPU/GPU)
        epoch: номер эпохи (для логов)
    
    Возвращает:
        avg_loss, avg_accuracy
    """
    
    # Переводим модель в режим обучения
    # Это включает Dropout и BatchNorm в режим обучения
    model.train()
    
    # Счетчики
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Прогресс-бар
    progress_bar = tqdm(dataloader, desc=f"TRAIN EPOCH {epoch}")
    
    for images, labels in progress_bar:
        # 1. Переносим данные на устройство (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # 2. Обнуляем градиенты от предыдущего шага
        optimizer.zero_grad()
        
        # 3. Forward pass: прогоняем данные через модель
        logits = model(images)
        
        # 4. Вычисляем loss
        loss = loss_fn(logits, labels)
        
        # 5. Backward pass: вычисляем градиенты
        loss.backward()
        
        # 6. Обновляем веса
        optimizer.step()
        
        # 7. Собираем статистику
        total_loss += loss.item() * images.size(0)
        
        # Предсказания: берем класс с максимальным логитом
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        
        # Обновляем прогресс-бар
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{total_correct / total_samples:.4f}",
        })
        
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
        
    return avg_loss, avg_accuracy
    
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Оценивает модель на валидации.
    
    Отличия от train:
    - model.eval() вместо model.train()
    - torch.no_grad() - не вычисляем градиенты
    """
    
    # Переводим модель в режим оценки
    # Это отключает Dropout и переводит BatchNorm в режим inference
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Отключаем вычисление градиентов (экономит память и время)
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='VALIDATION')
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = loss_fn(logits, labels)
            
            # Статистика
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{total_correct / total_samples:.4f}",
            })
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_correct / total_samples
    
    return avg_loss, avg_accuracy