import cv2
import pickle
import time
import os
from datetime import datetime
from ultralytics import YOLO
from utils import get_model_short_name, create_logger, expand_bbox, extract_features_from_crop


def run_inference(config):
    """
    Запускает инференс (детекция + классификация эмбриона) на видео или папке с изображениями.

    Аргументы:
        config (dict): словарь конфигурации со следующими ключами
            detection_model_path (str): путь к файлу .pt модели детекции яиц
            classifier_path (str): путь к файлу .pkl обученного классификатора SVM
            source (str): путь к видеофайлу, число (веб-камера) или путь к папке с изображениями
            output (str): путь для сохранения видео (для папки не используется)

            conf_thresh (float, optional): порог уверенности детектора (по умолч. 0.75)
            iou_thresh (float, optional): порог IoU для NMS (по умолч. 0.6)
            device (str, optional): 'cuda' или 'cpu' (по умолч. 'cuda')

            expand_ratio (float, optional): расширение bounding box (по умолч. 0.1)
            median_blur (int, optional): размер ядра медианного фильтра (по умолч. 3)
            val_thresh (int, optional): порог яркости для маски (по умолч. 45)
            sat_thresh (int, optional): порог насыщенности для маски (по умолч. 25)
            hist_bins (int, optional): количество бинов гистограммы Hue (по умолч. 32)
            hist_range (tuple, optional): диапазон Hue (по умолч. (0, 180))

            threshold (float, optional): порог вероятности для классификации (по умолч. 0.5)
            color_embryo (tuple, optional): цвет рамки для эмбриона (BGR) (по умолч. (0,255,0))
            color_empty (tuple, optional): цвет рамки для пустого яйца (BGR) (по умолч. (0,0,255))
            log_enabled (bool, optional): записывать лог в папку inference_logs (по умолч. True)
            show_fps (bool, optional): отображать текущий FPS на видео (по умолч. True)

    Примечания:
        Параметры извлечения признаков (expand_ratio, median_blur, val_thresh, sat_thresh,
        hist_bins, hist_range) должны быть строго теми же, что использовались при обучении
        классификатора, иначе векторы признаков будут несовместимы.
    """
    detection_model_path = config['detection_model_path']
    classifier_path = config['classifier_path']
    source = config['source']
    output = config.get('output', None)

    conf_thresh = config.get('conf_thresh', 0.75)
    iou_thresh = config.get('iou_thresh', 0.6)
    device = config.get('device', 'cuda')

    expand_ratio = config.get('expand_ratio', 0.1)
    median_blur = config.get('median_blur', 3)
    val_thresh = config.get('val_thresh', 45)
    sat_thresh = config.get('sat_thresh', 25)
    hist_bins = config.get('hist_bins', 32)
    hist_range = config.get('hist_range', (0, 180))

    threshold = config.get('threshold', 0.5)
    color_embryo = config.get('color_embryo', (0, 255, 0))
    color_empty = config.get('color_empty', (0, 0, 255))
    log_enabled = config.get('log_enabled', True)
    show_fps = config.get('show_fps', True)

    print("Загрузка детектора YOLO...")
    detector = YOLO(detection_model_path)
    print("Загрузка классификатора...")
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)

    if log_enabled:
        model_short = get_model_short_name(detection_model_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"inf_{timestamp}_{model_short}"
        log_write, run_dir, log_file = create_logger('inference_logs', run_name)
        log_write("=" * 60)
        log_write(f"Запуск инференса: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_write(f"Детектор: {model_short}")
        log_write(f"Классификатор: {classifier_path}")
        log_write(f"Порог: {threshold}")
        log_write(f"Источник: {source}")
        if output:
            log_write(f"Выход: {output}")
    else:
        def log_write(msg, also_print=True):
            if also_print:
                print(msg)

    is_video = str(source).isdigit() or (isinstance(source, str) and source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')))
    is_folder = isinstance(source, str) and not is_video and os.path.isdir(source)

    if not (is_video or is_folder):
        log_write("Ошибка: источник должен быть видеофайлом, номером камеры или папкой с изображениями", also_print=True)
        return

    def process_frame(frame):
        results = detector(frame, conf=conf_thresh, iou=iou_thresh, device=device, verbose=False)
        egg_cnt = 0
        embryo_cnt = 0
        empty_cnt = 0
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                nx1, ny1, nx2, ny2 = expand_bbox(x1, y1, x2, y2, frame.shape, expand_ratio)
                crop = frame[ny1:ny2, nx1:nx2]
                if crop.size == 0:
                    continue
                features = extract_features_from_crop(crop, median_blur, val_thresh, sat_thresh, hist_bins, hist_range)
                if features is not None:
                    proba = classifier.predict_proba([features])[0][1]
                    is_embryo = proba >= threshold
                    label = f"Embryo ({proba:.2f})" if is_embryo else f"Empty ({1-proba:.2f})"
                    color = color_embryo if is_embryo else color_empty
                    egg_cnt += 1
                    if is_embryo:
                        embryo_cnt += 1
                    else:
                        empty_cnt += 1
                    cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), color, 2)
                    cv2.putText(frame, label, (nx1, ny1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame, egg_cnt, embryo_cnt, empty_cnt

    if is_video:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log_write("Ошибка: не удалось открыть видео", also_print=True)
            return
        fps_src = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (width, height))

        frame_count = 0
        total_eggs = 0
        total_embryo = 0
        total_empty = 0
        prev_time = time.time()
        start_time = prev_time

        log_write("Начинаем обработку видео...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame, egg_cnt, embryo_cnt, empty_cnt = process_frame(frame)
            if show_fps:
                current_time = time.time()
                dt = current_time - prev_time
                if dt > 0:
                    fps_display = 1.0 / dt
                else:
                    fps_display = 0.0
                prev_time = current_time
                cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            out.write(frame)
            total_eggs += egg_cnt
            total_embryo += embryo_cnt
            total_empty += empty_cnt
            if frame_count % 100 == 0:
                log_write(f"Обработано кадров: {frame_count}")
        cap.release()
        out.release()
        elapsed = time.time() - start_time
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        log_write(f"\nОбработка видео завершена. Всего кадров: {frame_count}, время: {elapsed:.2f} сек, средний FPS: {fps_actual:.2f}")
        log_write(f"Обнаружено яиц: {total_eggs} (эмбрионов: {total_embryo}, пустых: {total_empty})")
        log_write(f"Результат сохранён в {output}")

    else:
        supported_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        images = [f for f in os.listdir(source) if f.lower().endswith(supported_ext)]
        if not images:
            log_write(f"В папке {source} нет поддерживаемых изображений", also_print=True)
            return

        out_folder = source.rstrip(os.sep) + "_out"
        os.makedirs(out_folder, exist_ok=True)
        log_write(f"Создана выходная папка: {out_folder}")

        total_processed = 0
        total_eggs = 0
        total_embryo = 0
        total_empty = 0

        log_write("Начинаем обработку изображений...")
        for img_name in images:
            img_path = os.path.join(source, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                log_write(f"Не удалось прочитать {img_path}", also_print=True)
                continue
            frame, egg_cnt, embryo_cnt, empty_cnt = process_frame(frame)
            out_path = os.path.join(out_folder, img_name)
            cv2.imwrite(out_path, frame)
            total_processed += 1
            total_eggs += egg_cnt
            total_embryo += embryo_cnt
            total_empty += empty_cnt
            log_write(f"{img_name}: яиц {egg_cnt} (эмбрионов {embryo_cnt})")
        log_write(f"Обработано изображений: {total_processed}")
        log_write(f"Всего обнаружено яиц: {total_eggs} (эмбрионов: {total_embryo}, пустых: {total_empty})")
        log_write(f"Результаты сохранены в {out_folder}")

    if log_enabled:
        log_write("=" * 60)


if __name__ == '__main__':
    # config = {
    #     'detection_model_path': 'runs/detect/egg_detector/yolov8n_eggs-7/weights/last.pt',
    #     'classifier_path': 'classifier_logs/cl_20260615_041728_yolov8n_eggs-7-last/classifier.pkl',
    #     'source': 'test_video.mp4',
    #     'output': 'output_classified.mp4',
    #     'conf_thresh': 0.75,
    #     'iou_thresh': 0.6,
    #     'device': 'cuda',
    #     'expand_ratio': 0.0,
    #     'median_blur': 3,
    #     'val_thresh': 50,
    #     'sat_thresh': 30,
    #     'hist_bins': 32,
    #     'hist_range': (0, 180),
    #     'threshold': 0.7845,
    #     'color_embryo': (0, 255, 0),
    #     'color_empty': (0, 0, 255),
    #     'log_enabled': True,
    #     'show_fps': True
    # }
    config = {
        'detection_model_path': 'runs/detect/egg_detector/yolov8n_eggs-7/weights/last.pt',
        'classifier_path': 'classifier_logs/cl_20260615_041728_yolov8n_eggs-7-last/classifier.pkl',
        'source': 'embryo_test',
        'conf_thresh': 0.75,
        'iou_thresh': 0.6,
        'device': 'cuda',
        'expand_ratio': 0.0,
        'median_blur': 3,
        'val_thresh': 50,
        'sat_thresh': 30,
        'hist_bins': 32,
        'hist_range': (0, 180),
        'threshold': 0.7845,
        'color_embryo': (0, 255, 0),
        'color_empty': (0, 0, 255),
        'log_enabled': True,
        'show_fps': False
    }

    run_inference(config)
