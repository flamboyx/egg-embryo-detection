import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ultralytics import YOLO

DETECTION_MODEL_PATH = 'runs/detect/egg_detector/yolov8n_eggs-7/weights/last.pt'
DATA_ROOT = 'embryo_classification_data'
FERTILE_DIR = os.path.join(DATA_ROOT, 'fertile')
INFERTILE_DIR = os.path.join(DATA_ROOT, 'infertile')

CONF_THRESH = 0.75
IOU_THRESH = 0.6
DEVICE = 'cuda'

EXPAND_RATIO = 0.2          # расширение bounding box на 10%
MEDIAN_BLUR = 3             # размер ядра медианного фильтра (0 = отключить)
VAL_THRESH = 30             # минимальная яркость для маски
SAT_THRESH = 20             # минимальная насыщенность для маски
HIST_BINS = 32              # количество бинов гистограммы Hue
HIST_RANGE = (0, 180)       # диапазон Hue

TEST_SIZE = 0.2
RANDOM_STATE = 42

print("Загрузка модели детекции...")
detector = YOLO(DETECTION_MODEL_PATH)


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

    # Маска ярких и насыщенных пикселей (яйцо / воздушная камера)
    mask = (val > VAL_THRESH) & (sat > SAT_THRESH)
    total_pixels = crop.shape[0] * crop.shape[1]
    bright_ratio = np.sum(mask) / total_pixels if total_pixels > 0 else 0.0

    # Гистограмма Hue по маске
    if np.sum(mask) == 0:
        hist = np.zeros(HIST_BINS)
    else:
        hist = cv2.calcHist([hue], [0], mask.astype(np.uint8), [HIST_BINS], HIST_RANGE)
        hist = hist / (hist.sum() + 1e-7)

    # Дополнительные признаки
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
                print(f"Предупреждение: яйцо не найдено на {full_path}")
                continue
            feats = extract_features_from_crop(crop)
            if feats is not None:
                features.append(feats)
                labels.append(label)
    return features, labels


print("\nОбработка фертильных яиц...")
fert_features, fert_labels = process_folder(FERTILE_DIR, 1)
print(f"Фертильных: {len(fert_features)}")

print("Обработка инфертильных яиц...")
inf_features, inf_labels = process_folder(INFERTILE_DIR, 0)
print(f"Инфертильных: {len(inf_features)}")

if len(fert_features) == 0 or len(inf_features) == 0:
    print("Ошибка: недостаточно данных. Проверьте пути.")
    exit()

X = np.vstack([fert_features, inf_features])
y = np.hstack([fert_labels, inf_labels])
print(f"Всего образцов: {len(X)} (фертильных={sum(y==1)}, инфертильных={sum(y==0)})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"Обучающая выборка: {len(X_train)} (ферт={sum(y_train==1)}, инферт={sum(y_train==0)})")
print(f"Тестовая выборка: {len(X_test)} (ферт={sum(y_test==1)}, инферт={sum(y_test==0)})")

print("\nОбучение SVM...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nРЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Confusion matrix:")
print(cm)

with open('embryo_classifier.pkl', 'wb') as f:
    pickle.dump(svm, f)
print("\nКлассификатор сохранён в 'embryo_classifier.pkl'")