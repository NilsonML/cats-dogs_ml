"""
data_setup.py - Создает DataLoader'ы для train и val.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_transforms(image_size: int = 224):
    """
    Создает трансформации для train и val.
    
    Аргументы:
        image_size: размер изображения на выходе (224 для ResNet)
    
    Возвращает:
        train_transforms, val_transforms
    
    Почему разные трансформации:
    - Для train используем аугментации (случайные изменения) → модель учится обобщать
    - Для val только нормализация → стабильная оценка качества
    """
    
    # Стандартные значения для моделей, предобученных на ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Трансформации для TRAIN (с аугментациями)
    train_transforms = transforms.Compose([
        # 1. Изменяем размер до 256 (чуть больше целевого)
        transforms.Resize((image_size + 32, image_size + 32)),
        
        # 2. Случайно обрезаем до 224x224
        #    Это дает разные части изображения → разнообразие
        transforms.RandomCrop((image_size, image_size)),
        
        # 3. Случайно отражаем по горизонтали (50% вероятность)
        #    Кошка слева = кошка справа, модель должна понять
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 4. Случайно поворачиваем на ±15 градусов
        transforms.RandomRotation(degrees=15),
        
        # 5. Слегка меняем яркость, контраст, насыщенность
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        # 6. Преобразуем в тензор [0, 1]
        transforms.ToTensor(),
        
        # 7. Нормализуем: (x - mean) / std
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # Трансформации для VAL (без аугментаций!)
    # Тот же масштаб, что и в train: Resize до 256 и центральный кроп 224.
    # Если на val просто сжать до 224, объекты будут крупнее, чем модель
    # видела при обучении (там кроп 224 из 256) — лишний сдвиг распределения.
    val_transforms = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return train_transforms, val_transforms

def create_dataloader(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
):
    """
    Создает DataLoader'ы для train и val.
    
    Аргументы:
        train_dir: путь к папке с train
        val_dir: путь к папке с val
        batch_size: сколько изображений в одном батче
        num_workers: сколько процессов для загрузки данных
        image_size: размер изображения
    
    Возвращает:
        train_loader, val_loader, class_names
    """
    
    # Создаем трансформации
    train_transforms, val_transforms = create_transforms(image_size)
    
    # Создаем датасеты
    # ImageFolder автоматически найдет классы по именам папок
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    # Создаем DataLoader'ы
    # DataLoader разбивает данные на батчи и перемешивает их
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # pin_memory полезен только при копировании на GPU
        )
    
    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    
    class_names = train_dataset.classes
    
    return train_loader, val_loader, class_names

