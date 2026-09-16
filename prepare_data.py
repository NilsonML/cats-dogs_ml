"""
prepare_data.py - Разделяет датасет на train/val с правильной структурой папок.
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Пути
SOURCE_DIR = Path("data/train_original")  # Папка с исходными изображениями
TARGET_DIR = Path("data")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Фиксируем seed для воспроизводимости
random.seed(RANDOM_SEED)

def create_dirs():
    """Создает структуру папок для train и val."""
    for split in ["train", "val"]:
        for class_name in ["cats", "dogs"]:
            (TARGET_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    print("✅ Папки созданы")

def split_data():
    """
    Разделяет изображения на train и val.
    
    Логика:
    - Собираем все файлы кошек и собак
    - Перемешиваем
    - 80% → train, 20% → val
    - Копируем файлы в соответствующие папки
    """
    # Собираем файлы
    cat_files = [f for f in os.listdir(SOURCE_DIR) if f.startswith("cat")]
    dog_files = [f for f in os.listdir(SOURCE_DIR) if f.startswith("dog")]
    
    # Перемешиваем
    random.shuffle(cat_files)
    random.shuffle(dog_files)
    
    # Разделяем
    cat_split = int(len(cat_files) * TRAIN_RATIO)
    dog_split = int(len(dog_files) * TRAIN_RATIO)
    
    train_cats = cat_files[:cat_split]
    val_cats = cat_files[cat_split:]
    train_dogs = dog_files[:dog_split]
    val_dogs = dog_files[dog_split:]
    
    print(f"📊 Train: {len(train_cats)} cats, {len(train_dogs)} dogs")
    print(f"📊 Val: {len(val_cats)} cats, {len(val_dogs)} dogs")
    
    # Копируем файлы
    def copy_files(files, src, dst):
        for f in tqdm(files, desc=f"Копирование в {dst}"):
            shutil.copy(src / f, dst / f)
    
    copy_files(train_cats, SOURCE_DIR, TARGET_DIR / "train" / "cats")
    copy_files(val_cats, SOURCE_DIR, TARGET_DIR / "val" / "cats")
    copy_files(train_dogs, SOURCE_DIR, TARGET_DIR / "train" / "dogs")
    copy_files(val_dogs, SOURCE_DIR, TARGET_DIR / "val" / "dogs")
    
    print("✅ Данные разделены!")

if __name__ == "__main__":
    create_dirs()
    split_data()