import os


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