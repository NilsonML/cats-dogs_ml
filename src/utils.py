"""
utils.py - Вспомогательные функции.

- save_model: сохранение модели
- load_model: загрузка модели
- save_metrics: сохранение метрик в CSV
"""

import torch
from pathlib import Path
import csv

def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Сохраняет модель в файл.
    
    Аргументы:
        model: модель для сохранения
        target_dir: папка для сохранения
        model_name: имя файла (без расширения)
    """
    
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Формируем имя файла: model_name.pth
    model_path = target_dir / f"{model_name}.pth"
    
    # Сохраняем state_dict (веса модели)
    # state_dict - это словарь {имя_параметра: тензор}
    torch.save(model.state_dict(),model_path)
    
    print(f"💾 Модель сохранена: {model_path}")
    
def load_model(model: torch.nn.Module, model_path: str, device: torch.device):
    """Загружает веса в модель."""
    # weights_only=True: загружаем только тензоры. Без него torch.load
    # распаковывает произвольные pickle-объекты, а это исполнение чужого кода.
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"✅ Модель загружена: {model_path}")
    return model

def save_metrics(metrics: dict, filepath: str):
    """
    Сохраняет метрики в CSV.
    
    Аргументы:
        metrics: словарь {'epoch': [1,2,3], 'train_loss': [...], ...}
        filepath: путь к файлу
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Получаем все ключи (названия колонок)
    keys = list(metrics.keys())
    
    # Количество эпох
    n_epochs = len(metrics[keys[0]])
    
    # Записываем в CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        
        for i in range(n_epochs):
            row = [metrics[k][i] for k in keys]
            writer.writerow(row)
    
    print(f"📊 Метрики сохранены: {filepath}")
    