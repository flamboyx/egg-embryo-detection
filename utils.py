import cv2
import numpy as np
import os


def create_logger(log_dir, run_name):
    """
    Создаёт папку для логов и возвращает функцию log_write.
    Аргументы:
        log_dir (str): корневая папка для логов
        run_name (str): имя подпапки
    Возвращает:
        log_write: функция, принимающая msg (str) и also_print (bool, по умолч. True)
        run_dir: полный путь к созданной папке
        log_file: полный путь к файлу лога
    """
    run_dir = os.path.join(log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, 'log.txt')

    def log_write(msg, also_print=True):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        if also_print:
            print(msg)

    return log_write, run_dir, log_file


def expand_bbox(x1, y1, x2, y2, img_shape, expand_ratio=0.1):
    """
    Расширяет bounding box на заданный коэффициент, не выходя за границы изображения.

    Аргументы:
        x1, y1 (int): координаты верхнего левого угла исходного прямоугольника
        x2, y2 (int): координаты нижнего правого угла исходного прямоугольника
        img_shape (tuple): форма изображения
        expand_ratio (float): коэффициент расширения (0.1 = 10%)

    Возвращает:
        tuple: (nx1, ny1, nx2, ny2) – новые координаты расширенного прямоугольника
    """
    h, w = img_shape[:2]
    dx = int((x2 - x1) * expand_ratio)
    dy = int((y2 - y1) * expand_ratio)
    nx1 = max(0, x1 - dx)
    nx2 = min(w, x2 + dx)
    ny1 = max(0, y1 - dy)
    ny2 = min(h, y2 + dy)

    return nx1, ny1, nx2, ny2


def extract_features_from_crop(crop, median_blur, val_thresh, sat_thresh, hist_bins, hist_range):
    """
    Извлекает вектор признаков из кропа яйца для классификации эмбриона.

    Аргументы:
        crop (numpy.ndarray): изображение кропа
        median_blur (int): размер ядра медианного фильтра (0 – отключить)
        val_thresh (int): порог яркости (Value) для маски
        sat_thresh (int): порог насыщенности (Saturation) для маски
        hist_bins (int): количество бинов гистограммы Hue
        hist_range (tuple): диапазон Hue (min, max)

    Возвращает:
        numpy.ndarray: вектор признаков
    """
    if crop is None or crop.size == 0:
        return None
    if median_blur > 0:
        crop = cv2.medianBlur(crop, median_blur)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (val > val_thresh) & (sat > sat_thresh)
    total_pixels = crop.shape[0] * crop.shape[1]
    bright_ratio = np.sum(mask) / total_pixels if total_pixels > 0 else 0.0
    if np.sum(mask) == 0:
        hist = np.zeros(hist_bins)
    else:
        hist = cv2.calcHist([hue], [0], mask.astype(np.uint8), [hist_bins], hist_range)
        hist = hist / (hist.sum() + 1e-7)
    mean_val = np.mean(val)
    mean_sat = np.mean(sat)
    features = np.hstack([hist.flatten(), [bright_ratio, mean_val, mean_sat]])

    return features.astype(np.float32)


def get_model_short_name(model_path):
    """
    Извлекает короткое имя модели из полного пути.
    Пример: runs/detect/egg_detector/yolov8n_eggs-7/weights/last.pt -> yolov8n_eggs-7-last
    """
    norm_path = os.path.normpath(model_path)
    parts = norm_path.split(os.sep)
    try:
        weights_idx = parts.index('weights')
        model_name = parts[weights_idx - 1]
    except ValueError:
        model_name = os.path.basename(os.path.dirname(model_path))
    base_name = os.path.splitext(os.path.basename(model_path))[0]

    return f"{model_name}-{base_name}"
