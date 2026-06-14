import os
import cv2
import numpy as np
import pickle
import json
import shutil
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from ultralytics import YOLO
from utils import get_model_short_name, create_logger, expand_bbox, extract_features_from_crop


def extract_egg_crop(detector, image_path, conf_thresh, iou_thresh, device, expand_ratio):
    """Находит яйцо на изображении, возвращает кроп с расширением."""
    results = detector(image_path, conf=conf_thresh, iou=iou_thresh, device=device, verbose=False)
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
    nx1, ny1, nx2, ny2 = expand_bbox(x1, y1, x2, y2, img.shape, expand_ratio)

    return img[ny1:ny2, nx1:nx2]


def process_folder(detector, folder_path, label, log_write, params):
    """Обрабатывает все изображения в папке, возвращает признаки и метки."""
    features = []
    labels = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(folder_path, fname)
            crop = extract_egg_crop(detector, full_path,
                                    params['conf_thresh'], params['iou_thresh'],
                                    params['device'], params['expand_ratio'])
            if crop is None:
                log_write(f"Предупреждение: яйцо не найдено на {full_path}")
                continue
            feats = extract_features_from_crop(crop,
                                               params['median_blur'], params['val_thresh'],
                                               params['sat_thresh'], params['hist_bins'],
                                               params['hist_range'])
            if feats is not None:
                features.append(feats)
                labels.append(label)
    return features, labels


def augment_class(src_dir, target_count, class_name, data_root, augment_params):
    """Создаёт аугментированные копии изображений в src_dir, чтобы достичь target_count."""
    aug_dir = os.path.join(data_root, f'{class_name}_augmented_temp')
    os.makedirs(aug_dir, exist_ok=True)
    orig_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in orig_files:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(aug_dir, fname))
    current_count = len(orig_files)
    if current_count >= target_count:
        print(f"  {class_name}: уже {current_count} (>= {target_count}), аугментация не требуется")
        return aug_dir, current_count

    angles = augment_params.get('angles', [-5, -2, 2, 5])
    scales = augment_params.get('scales', [0.95, 1.05])
    brightness = augment_params.get('brightness', [-15, 15])
    flips = augment_params.get('flips', [True, False])

    for fname in orig_files:
        img = cv2.imread(os.path.join(src_dir, fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        for ang in angles:
            for sc in scales:
                for b in brightness:
                    for flip in flips:
                        if current_count >= target_count:
                            break
                        M = cv2.getRotationMatrix2D((w/2, h/2), ang, sc)
                        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                        if b != 0:
                            rotated = cv2.convertScaleAbs(rotated, alpha=1.0, beta=b)
                        if flip:
                            rotated = cv2.flip(rotated, 1)
                        out_name = f"{os.path.splitext(fname)[0]}_aug_{ang}_{sc}_{b}_{flip}.jpg"
                        out_path = os.path.join(aug_dir, out_name)
                        cv2.imwrite(out_path, rotated)
                        current_count += 1
                    if current_count >= target_count:
                        break
                if current_count >= target_count:
                    break
            if current_count >= target_count:
                break
        if current_count >= target_count:
            break
    print(f"  {class_name}: после аугментации {current_count} изображений")

    return aug_dir, current_count


def train_embryo_classifier(config):
    """
    Обучение классификатора эмбриона на основе цветовых признаков.

    Аргументы:
        config (dict): словарь конфигурации со следующими ключами
            detection_model_path (str): путь к файлу .pt модели детекции яиц
            data_root (str): корневая папка с подпапками fertile и infertile
            features_params (dict): параметры извлечения признаков
                conf_thresh (float): порог уверенности детектора
                iou_thresh (float): порог IoU для NMS
                device (str): 'cuda' или 'cpu'
                expand_ratio (float): расширение bounding box (0.1 = 10%)
                median_blur (int): размер ядра медианного фильтра (0 = отключить)
                val_thresh (int): минимальная яркость для маски
                sat_thresh (int): минимальная насыщенность для маски
                hist_bins (int): количество бинов гистограммы Hue
                hist_range (tuple): диапазон Hue (по умолчанию (0, 180))
            test_size (float, optional): доля тестовой выборки (по умолчанию 0.2)
            random_state (int, optional): seed для воспроизводимости (по умолчанию 42)
            param_grid (dict, optional): сетка гиперпараметров для GridSearchCV (C, gamma)
                C (list of float): параметр регуляризации
                gamma (list): коэффициент ядра RBF
            augment_config (dict, optional): настройки аугментации
                apply_to (str): 'fertile', 'infertile', 'both' или 'none'
                target_count (int): целевое количество изображений после аугментации
                angles (list): углы поворота в градусах
                scales (list): коэффициенты масштабирования
                brightness (list): смещения яркости
                flips (list): флаги горизонтального отражения

    Возвращает:
        tuple: (best_svm, metrics_dict), где
            best_svm (SVC): обученная модель SVM
            metrics_dict (dict): словарь с метриками, параметрами и путями к сохранённым файлам
    """
    detection_model_path = config['detection_model_path']
    data_root = config['data_root']
    fertile_dir = os.path.join(data_root, 'fertile')
    infertile_dir = os.path.join(data_root, 'infertile')

    params = config['features_params']
    test_size = config.get('test_size', 0.2)
    random_state = config.get('random_state', 42)
    param_grid = config.get('param_grid', {
        'C': [0.03, 0.1, 0.3, 1, 3, 10],
        'gamma': [0.003, 0.01, 0.03, 0.1, 0.3, 1, 'scale', 'auto']
    })
    augment_config = config.get('augment_config', None)  # может быть None

    model_short = get_model_short_name(detection_model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"cl_{timestamp}_{model_short}"
    log_write, run_dir, log_file = create_logger('classifier_logs', run_name)
    json_file = os.path.join(run_dir, 'metrics.json')
    model_save_path = os.path.join(run_dir, 'classifier.pkl')

    log_write("=" * 60)
    log_write(f"Запуск обучения классификатора: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_write(f"Модель детекции: {model_short}")
    log_write("Параметры извлечения признаков:")
    for k, v in params.items():
        log_write(f"  {k} = {v}")
    log_write(f"test_size = {test_size}")
    log_write(f"random_state = {random_state}")
    log_write(f"param_grid = {param_grid}")
    if augment_config:
        log_write(f"augment_config = {augment_config}")

    temp_dirs_to_clean = []
    fertile_used_dir = fertile_dir
    infertile_used_dir = infertile_dir

    if augment_config:
        apply_to = augment_config.get('apply_to', 'none')
        target = augment_config.get('target_count', 400)
        augment_params = {k: v for k, v in augment_config.items() if k not in ['apply_to', 'target_count']}
        if apply_to in ['fertile', 'both']:
            log_write("\nАугментация оплодотворённых яиц...")
            fert_aug_dir, fert_count = augment_class(fertile_dir, target, 'fertile', data_root, augment_params)
            fertile_used_dir = fert_aug_dir
            temp_dirs_to_clean.append(fert_aug_dir)
        else:
            fert_count = len([f for f in os.listdir(fertile_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if apply_to in ['infertile', 'both']:
            log_write("\nАугментация неоплодотворённых яиц...")
            inf_aug_dir, inf_count = augment_class(infertile_dir, target, 'infertile', data_root, augment_params)
            infertile_used_dir = inf_aug_dir
            temp_dirs_to_clean.append(inf_aug_dir)
        else:
            inf_count = len([f for f in os.listdir(infertile_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        log_write(f"Итоговое количество: оплодотворённых = {fert_count}, неоплодотворённых = {inf_count}")
    else:
        fert_count = len([f for f in os.listdir(fertile_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        inf_count = len([f for f in os.listdir(infertile_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        log_write(f"Количество (без аугментации): оплодотворённых = {fert_count}, неоплодотворённых = {inf_count}")

    log_write("\nЗагрузка модели детекции...")
    detector = YOLO(detection_model_path)

    log_write("\nОбработка оплодотворённых яиц...")
    fert_features, fert_labels = process_folder(detector, fertile_used_dir, 1, log_write, params)
    log_write(f"Оплодотворённых (успешно извлечено): {len(fert_features)}")

    log_write("Обработка неоплодотворённых яиц...")
    inf_features, inf_labels = process_folder(detector, infertile_used_dir, 0, log_write, params)
    log_write(f"Неоплодотворённых (успешно извлечено): {len(inf_features)}")

    if len(fert_features) == 0 or len(inf_features) == 0:
        log_write("Ошибка: недостаточно данных. Проверьте пути.")
        return

    X = np.vstack([fert_features, inf_features])
    y = np.hstack([fert_labels, inf_labels])
    log_write(f"Всего образцов: {len(X)} (оплодотворённых={sum(y==1)}, неоплодотворённых={sum(y==0)})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    log_write(f"Обучающая выборка: {len(X_train)} (опл={sum(y_train==1)}, неопл={sum(y_train==0)})")
    log_write(f"Тестовая выборка: {len(X_test)} (опл={sum(y_test==1)}, неопл={sum(y_test==0)})")

    log_write("\nПоиск оптимальных параметров SVM (GridSearchCV)...")
    base_svm = SVC(class_weight='balanced', probability=True, random_state=random_state)
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

    y_proba = best_svm.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
    log_write(f"\nROC AUC: {roc_auc:.4f}")
    log_write(f"Оптимальный порог (Юден): {optimal_threshold:.4f}")

    with open(model_save_path, 'wb') as f:
        pickle.dump(best_svm, f)
    log_write(f"\nЛучший классификатор сохранён в {model_save_path}")

    metrics_dict = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detection_model': model_short,
        'augmentation_config': augment_config,
        'parameters': params,
        'test_size': test_size,
        'random_state': random_state,
        'param_grid': param_grid,
        'best_params': best_params,
        'optimal_threshold': optimal_threshold,
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
            'best_cv_f1': grid.best_score_,
            'roc_auc': roc_auc
        }
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    log_write(f"Метрики сохранены в {json_file}")

    for temp_dir in temp_dirs_to_clean:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            log_write(f"Удалена временная папка: {temp_dir}")

    log_write("=" * 60)

    return best_svm, metrics_dict


if __name__ == '__main__':
    config = {
        'detection_model_path': 'runs/detect/egg_detector/yolov8n_eggs-7/weights/last.pt',
        'data_root': 'embryo_classification_data',
        'features_params': {
            'conf_thresh': 0.75,
            'iou_thresh': 0.6,
            'device': 'cuda',
            'expand_ratio': 0.0,
            'median_blur': 3,
            'val_thresh': 50,
            'sat_thresh': 30,
            'hist_bins': 32,
            'hist_range': (0, 180)
        },
        'test_size': 0.2,
        'random_state': 42,
        'param_grid': {
            'C': [0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10],
            'gamma': [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 'scale', 'auto']
        },
        'augment_config': {
            'apply_to': 'both',
            'target_count': 500,
            'angles': [-10, -5, 5, 10],
            'scales': [0.95, 1.05],
            'brightness': [-10, 10],
            'flips': [True, False]
        }
    }

    train_embryo_classifier(config)
