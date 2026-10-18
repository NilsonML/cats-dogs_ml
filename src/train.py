"""
train.py - Главный скрипт для обучения.

Запуск (из корня репозитория):
    python -m src.train
    python -m src.train --epochs 3 --batch-size 64
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from src.data_setup import create_dataloader
from src.engine import evaluate, train_one_epoch
from src.model import create_model
from src.utils import save_metrics, save_model, set_seed

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
CONFIG = {
    "train_dir": "data/train",
    # Было "data/test": такой папки prepare_data.py не создаёт (он делит на train/val),
    # и обучение падало с FileNotFoundError на первом же шаге.
    "val_dir": "data/val",
    "batch_size": 32,
    "num_workers": 4,
    "image_size": 224,
    "model_name": "resnet18",
    "num_classes": 2,
    "freeze_features": True,  # Сначала обучаем только последний слой
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "save_dir": "models",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def parse_args() -> argparse.Namespace:
    """Переопределение самых частых параметров из командной строки,
    чтобы для эксперимента не приходилось править код."""
    parser = argparse.ArgumentParser(description="Cats vs Dogs: transfer learning на ResNet18")
    parser.add_argument("--epochs", type=int, default=CONFIG["num_epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["learning_rate"])
    parser.add_argument("--num-workers", type=int, default=CONFIG["num_workers"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    return parser.parse_args()


def main():
    """Главная функция обучения."""
    args = parse_args()
    CONFIG.update(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    set_seed(CONFIG["seed"])

    print("=" * 60)
    print("🚀 CATS VS DOGS - TRAINING")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Model: {CONFIG['model_name']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")

    # ============================================================
    # 1. СОЗДАЕМ DATALOADER'Ы
    # ============================================================
    print("\n📂 Загрузка данных...")
    train_loader, val_loader, class_names = create_dataloader(
        train_dir=CONFIG["train_dir"],
        val_dir=CONFIG["val_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        image_size=CONFIG["image_size"],
        pin_memory=CONFIG["device"] == "cuda",
    )

    print(f"✅ Классы: {class_names}")
    print(f"✅ Train батчей: {len(train_loader)}")
    print(f"✅ Val батчей: {len(val_loader)}")

    # ============================================================
    # 2. СОЗДАЕМ МОДЕЛЬ
    # ============================================================
    print("\n🧠 Создание модели...")
    model = create_model(
        model_name=CONFIG["model_name"],
        num_classes=CONFIG["num_classes"],
        freeze_features=CONFIG["freeze_features"],
    )
    model = model.to(CONFIG["device"])

    # Считаем количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Всего параметров: {total_params:,}")
    print(f"✅ Обучаемых параметров: {trainable_params:,}")

    # ============================================================
    # 3. ОПРЕДЕЛЯЕМ LOSS И ОПТИМИЗАТОР
    # ============================================================

    loss_fn = nn.CrossEntropyLoss()

    # Обучаем только параметры, которые требуют градиента
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ============================================================
    # 4. ЦИКЛ ОБУЧЕНИЯ
    # ============================================================
    print("\n" + "=" * 60)
    print("🏋️  НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)

    metrics = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        print(f"\n📅 Эпоха {epoch}/{CONFIG['num_epochs']}")
        print("-" * 40)

        # Обучение
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=CONFIG["device"],
            epoch=epoch,
        )

        # Оценка
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=CONFIG["device"],
        )

        # Логи
        print(f"\n📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Сохраняем метрики
        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, CONFIG["save_dir"], f"best_{CONFIG['model_name']}")
            print(f"🏆 Новая лучшая модель! Val Acc: {val_acc:.4f}")

    # ============================================================
    # 5. ИТОГИ
    # ============================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"⏱️  Время: {total_time / 60:.2f} минут")
    print(f"🏆 Лучшая Val Acc: {best_val_acc:.4f}")

    # Сохраняем метрики
    save_metrics(metrics, f"{CONFIG['save_dir']}/metrics.csv")


if __name__ == "__main__":
    main()
