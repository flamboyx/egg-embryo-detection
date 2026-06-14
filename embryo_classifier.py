import os
import cv2
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ultralytics import YOLO
from utils import get_model_short_name, create_logger


def extract_egg_crop(image_path):
    """Находит яйцо на изображении, возвращает кроп с расширением."""
    results = detector(image_path, conf=CONF_THRESH, iou=IOU_THRESH, device=DEVICE, verbose=False)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return None
    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    if len(confs) == 0:
        return None
    idx = np.argmax(confs)
    x1, y1, x2, y2 = map(int, boxes.xyxy[idx].cpu().numpy())
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    dx = int((x2 - x1) * EXPAND_RATIO)
    dy = int((y2 - y1) * EXPAND_RATIO)
    nx1 = max(0, x1 - dx)
    nx2 = min(w, x2 + dx)
    ny1 = max(0, y1 - dy)
    ny2 = min(h, y2 + dy)
    return img[ny1:ny2, nx1:nx2]


def extract_features_from_crop(crop):
    """Извлекает вектор признаков из кропа яйца."""
    if crop is None or crop.size == 0:
        return None
    if MEDIAN_BLUR > 0:
        crop = cv2.medianBlur(crop, MEDIAN_BLUR)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask = (val > VAL_THRESH) & (sat > SAT_THRESH)
    total_pixels = crop.shape[0] * crop.shape[1]
    bright_ratio = np.sum(mask) / total_pixels if total_pixels > 0 else 0.0

    if np.sum(mask) == 0:
        hist = np.zeros(HIST_BINS)
    else:
        hist = cv2.calcHist([hue], [0], mask.astype(np.uint8), [HIST_BINS], HIST_RANGE)
        hist = hist / (hist.sum() + 1e-7)

    mean_val = np.mean(val)
    mean_sat = np.mean(sat)

    features = np.hstack([hist.flatten(), [bright_ratio, mean_val, mean_sat]])
    return features.astype(np.float32)


def process_folder(folder_path, label):
    """Обрабатывает все изображения в папке, возвращает признаки и метки."""
    features = []
    labels = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, fname)
            crop = extract_egg_crop(full_path)
            if crop is None:
                log_write(f"Предупреждение: яйцо не найдено на {full_path}")
                continue
            feats = extract_features_from_crop(crop)
            if feats is not None:
                features.append(feats)
                labels.append(label)
    return features, labels


DETECTION_MODEL_PATH = 'runs/detect/egg_detector/yolov8n_eggs-7/weights/last.pt'
DATA_ROOT = 'embryo_classification_data'
FERTILE_DIR = os.path.join(DATA_ROOT, 'fertile')
INFERTILE_DIR = os.path.join(DATA_ROOT, 'infertile')

CONF_THRESH = 0.75
IOU_THRESH = 0.6
DEVICE = 'cuda'

EXPAND_RATIO = 0.1
MEDIAN_BLUR = 3
VAL_THRESH = 50
SAT_THRESH = 30
HIST_BINS = 32
HIST_RANGE = (0, 180)

TEST_SIZE = 0.2
RANDOM_STATE = 42

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

model_short = get_model_short_name(DETECTION_MODEL_PATH)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"cl_{timestamp}_{model_short}"
log_write, run_dir, log_file = create_logger('classifier_logs', run_name)
json_file = os.path.join(run_dir, 'metrics.json')
model_save_path = os.path.join(run_dir, 'classifier.pkl')

log_write("=" * 60)
log_write(f"Запуск обучения классификатора: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_write(f"Модель детекции: {model_short}")
log_write(f"Параметры:")
log_write(f"  CONF_THRESH = {CONF_THRESH}")
log_write(f"  IOU_THRESH = {IOU_THRESH}")
log_write(f"  DEVICE = {DEVICE}")
log_write(f"  EXPAND_RATIO = {EXPAND_RATIO}")
log_write(f"  MEDIAN_BLUR = {MEDIAN_BLUR}")
log_write(f"  VAL_THRESH = {VAL_THRESH}")
log_write(f"  SAT_THRESH = {SAT_THRESH}")
log_write(f"  HIST_BINS = {HIST_BINS}")
log_write(f"  HIST_RANGE = {HIST_RANGE}")
log_write(f"  TEST_SIZE = {TEST_SIZE}")
log_write(f"  RANDOM_STATE = {RANDOM_STATE}")
log_write(f"  param_grid = {param_grid}")

log_write("\nЗагрузка модели детекции...")
detector = YOLO(DETECTION_MODEL_PATH)

log_write("\nОбработка оплодотворённых яиц...")
fert_features, fert_labels = process_folder(FERTILE_DIR, 1)
log_write(f"Оплодотворённых: {len(fert_features)}")

log_write("Обработка неоплодотворённых яиц...")
inf_features, inf_labels = process_folder(INFERTILE_DIR, 0)
log_write(f"Неоплодотворённых: {len(inf_features)}")

if len(fert_features) == 0 or len(inf_features) == 0:
    log_write("Ошибка: недостаточно данных. Проверьте пути.")
    exit()

X = np.vstack([fert_features, inf_features])
y = np.hstack([fert_labels, inf_labels])
log_write(f"Всего образцов: {len(X)} (оплодотворённых={sum(y==1)}, неоплодотворённых={sum(y==0)})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
log_write(f"Обучающая выборка: {len(X_train)} (опл={sum(y_train==1)}, неопл={sum(y_train==0)})")
log_write(f"Тестовая выборка: {len(X_test)} (опл={sum(y_test==1)}, неопл={sum(y_test==0)})")

log_write("\nПоиск оптимальных параметров SVM (GridSearchCV)...")
base_svm = SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE)
grid = GridSearchCV(base_svm, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_svm = grid.best_estimator_
best_params = grid.best_params_
log_write(f"Лучшие параметры: {best_params}")
log_write(f"Лучшее CV F1-score: {grid.best_score_:.4f}")

y_pred = best_svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

log_write("\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ (на тестовой выборке)")
log_write(f"Accuracy:  {acc:.4f}")
log_write(f"Precision: {prec:.4f}")
log_write(f"Recall:    {rec:.4f}")
log_write(f"F1-score:  {f1:.4f}")
log_write(f"Confusion matrix:\n{cm}")

with open(model_save_path, 'wb') as f:
    pickle.dump(best_svm, f)
log_write(f"\nЛучший классификатор сохранён в {model_save_path}")

metrics_dict = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'detection_model': model_short,
    'parameters': {
        'conf_thresh': CONF_THRESH,
        'iou_thresh': IOU_THRESH,
        'device': DEVICE,
        'expand_ratio': EXPAND_RATIO,
        'median_blur': MEDIAN_BLUR,
        'val_thresh': VAL_THRESH,
        'sat_thresh': SAT_THRESH,
        'hist_bins': HIST_BINS,
        'hist_range': HIST_RANGE,
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'param_grid': param_grid,
        'best_params': best_params
    },
    'dataset': {
        'fertile': len(fert_features),
        'infertile': len(inf_features),
        'total': len(X),
        'train_size': len(X_train),
        'test_size': len(X_test)
    },
    'metrics': {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'best_cv_f1': grid.best_score_
    }
}

with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
log_write(f"Метрики сохранены в {json_file}")
log_write("=" * 60)
