import os
import textwrap
import time
import shutil
import torch
import cv2
import json
from datetime import datetime
from ultralytics import YOLO
from utils import get_model_short_name, create_logger


def test_model(model_path,
               dataset_path=None,
               video_path=None,
               imgsz=640,
               conf_threshold=0.5,
               iou_threshold=0.6,
               device='cuda',
               save=False):
    """
    Полное тестирование модели детекции.

    Аргументы:
        model_path (str): путь к файлу .pt
        dataset_path (str, optional): папка, где лежат изображения и txt-файлы (YOLO формат)
        video_path (str, optional): путь к видео
        imgsz (int): размер входного изображения
        conf_threshold (float): порог уверенности для детекции
        iou_threshold (float): порог NMS
        device (str): 'cuda' или 'cpu'
        save (bool): сохранять ли размеченные изображения и видео

    Возвращает:
        dict: словарь с метриками (precision, recall, mAP50, mAP50-95, F1, FPS_video, FPS_images)
    """

    model_short_name = get_model_short_name(model_path)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"test_{timestamp_str}_{model_short_name}"
    log_write, run_dir, log_file = create_logger('test_runs', run_name)

    log_write("=" * 60)
    log_write(f"Запуск тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_write(f"Модель: {model_short_name}")
    log_write(f"Параметры:")
    log_write(f"  imgsz: {imgsz}")
    log_write(f"  conf_threshold: {conf_threshold}")
    log_write(f"  iou_threshold: {iou_threshold}")
    log_write(f"  device: {device}")
    log_write(f"  save: {save}")

    if device == 'cuda' and not torch.cuda.is_available():
        log_write("CUDA недоступна, использую CPU")
        device = 'cpu'

    model = YOLO(model_path)

    metrics = {}

    if dataset_path is not None and os.path.exists(dataset_path):
        temp_dir = 'temp_test_data'
        os.makedirs(os.path.join(temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, 'labels'), exist_ok=True)

        for f in os.listdir(dataset_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(os.path.join(dataset_path, f), os.path.join(temp_dir, 'images', f))
                txt_name = os.path.splitext(f)[0] + '.txt'
                src_txt = os.path.join(dataset_path, txt_name)
                dst_txt = os.path.join(temp_dir, 'labels', txt_name)
                if os.path.exists(src_txt):
                    shutil.copy2(src_txt, dst_txt)
                else:
                    open(dst_txt, 'w').close()

        yaml_path = os.path.join(run_dir, 'test_dataset.yaml')
        yaml_content = textwrap.dedent(f"""
            path: {os.path.abspath(temp_dir)}
            train: images
            val: images
            test: images

            nc: 1
            names: ['egg']
        """)
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        log_write("\n=== Оценка качества на тестовых изображениях ===")
        results = model.val(
            data=yaml_path,
            batch=1,
            imgsz=imgsz,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            plots=True,
            save_json=True,
            save_hybrid=True,
            project='../../test_runs',
            name=run_name,
            exist_ok=True
        )

        precision = results.box.mp
        recall = results.box.mr
        mAP50 = results.box.map50
        mAP5095 = results.box.map
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        log_write(f"\nPrecision: {precision:.4f}")
        log_write(f"Recall: {recall:.4f}")
        log_write(f"mAP50: {mAP50:.4f}")
        log_write(f"mAP50-95: {mAP5095:.4f}")
        log_write(f"F1-score: {f1:.4f}")
        log_write(f"Результаты валидации сохранены в: {results.save_dir}")

        metrics.update({
            'precision': precision,
            'recall': recall,
            'mAP50': mAP50,
            'mAP50-95': mAP5095,
            'f1': f1,
        })

        log_write("\n=== Измерение FPS на фотографиях ===")
        image_files = [f for f in os.listdir(dataset_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) > 0:
            start_time = time.time()
            for img_file in image_files:
                img_path = os.path.join(dataset_path, img_file)
                pred = model(img_path, device=device, verbose=False)
                if save:
                    out_img_dir = os.path.join(run_dir, 'images')
                    os.makedirs(out_img_dir, exist_ok=True)
                    out_img_path = os.path.join(out_img_dir, img_file)
                    img = cv2.imread(img_path)
                    if pred[0].boxes is not None:
                        for box in pred[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.imwrite(out_img_path, img)
            elapsed = time.time() - start_time
            fps_images = len(image_files) / elapsed
            log_write(f"Обработано {len(image_files)} изображений за {elapsed:.2f} сек")
            log_write(f"FPS (на изображениях): {fps_images:.2f}")
            metrics['fps_images'] = fps_images
        else:
            log_write("Нет изображений для измерения FPS")
            metrics['fps_images'] = 0

        shutil.rmtree(temp_dir)
        if os.path.exists(yaml_path):
            os.remove(yaml_path)
    else:
        if dataset_path is not None:
            log_write(f"Внимание: папка {dataset_path} не найдена, пропускаем оценку на изображениях.")

    if video_path and os.path.exists(video_path):
        log_write(f"\n=== Измерение FPS на видео: {video_path} ===")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_write("Не удалось открыть видео")
            metrics['fps_video'] = 0
        else:
            frame_count = 0
            start_time = time.time()
            out_video = None
            if save:
                out_video_dir = os.path.join(run_dir, 'video')
                os.makedirs(out_video_dir, exist_ok=True)
                out_video_path = os.path.join(out_video_dir, 'detected_video.mp4')
                fps_vid_src = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(out_video_path, fourcc, fps_vid_src, (width, height))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results_frame = model(frame, device=device, verbose=False)
                if save and out_video:
                    annotated = results_frame[0].plot(line_width=1, labels=False, conf=False)
                    out_video.write(annotated)
                frame_count += 1
            elapsed = time.time() - start_time
            cap.release()
            if out_video:
                out_video.release()
                log_write(f"Размеченное видео сохранено в {out_video_path}")
            if elapsed > 0:
                fps_video = frame_count / elapsed
                log_write(f"Обработано {frame_count} кадров за {elapsed:.2f} сек")
                log_write(f"FPS (на видео): {fps_video:.2f}")
                metrics['fps_video'] = fps_video
            else:
                metrics['fps_video'] = 0
    else:
        if video_path:
            log_write(f"Видео не найдено: {video_path}")
        metrics['fps_video'] = 0

    json_metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_short_name,
        'parameters': {
            'imgsz': imgsz,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'device': device,
            'save': save,
        },
        'metrics': metrics
    }
    json_path = os.path.join(run_dir, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(json_metrics, jf, indent=4, ensure_ascii=False)
    log_write(f"JSON метрик сохранён: {json_path}")
    log_write("=" * 60)

    return metrics


if __name__ == '__main__':
    metrics = test_model(
        model_path='runs/detect/egg_detector/yolov8n_eggs-7/weights/last.pt',
        dataset_path='dataset_temp',
        video_path='test_video.mp4',
        conf_threshold=0.75,
        device='cuda',
        save=True
    )
