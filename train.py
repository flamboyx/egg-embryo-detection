from ultralytics import YOLO
import torch
import os


def get_unique_name(project, base_name):
    """Возвращает уникальное имя для эксперимента, добавляя номер версии."""
    exp_dir = os.path.join(project, base_name)
    version = 1
    new_name = base_name
    while os.path.exists(exp_dir):
        version += 1
        new_name = f"{base_name}_v{version}"
        exp_dir = os.path.join(project, new_name)
    return new_name


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    project = 'egg_detector'
    base_name = 'yolov8n_eggs'
    unique_name = get_unique_name(project, base_name)
    print(f"Имя эксперимента: {unique_name}")

    best_weights_path = 'runs/detect/egg_detector/yolov8n_eggs-6/weights/best.pt'
    model = YOLO(best_weights_path)
    # model = YOLO('yolov8n.pt')

    model.train(
        data='eggs_dataset/dataset.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        workers=0,
        device='cuda',
        project=project,
        name=unique_name,
        exist_ok=False,
        patience=20,
        plots=True,
        lr0=0.0001,
        lrf=0.001,
        weight_decay=0.0005,
        warmup_epochs=0,
        cos_lr=True,
    )

    print(f"Обучение завершено. Модель сохранена в '{project}/{unique_name}/weights/best.pt'")


if __name__ == '__main__':
    main()
