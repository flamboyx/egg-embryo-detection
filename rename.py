import os


def rename_images(folder_path, start_number=10001):
    files = os.listdir(folder_path)
    # Фильтруем только изображения по расширениям
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in files if f.lower().endswith(extensions)]
    # Сортируем для предсказуемого порядка (по имени)
    images.sort()

    for idx, old_name in enumerate(images):
        # Определяем расширение
        ext = os.path.splitext(old_name)[1]
        # Новое имя
        new_name = f"{start_number + idx}{ext}"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # Проверяем, не существует ли уже файл с новым именем
        if os.path.exists(new_path):
            print(f"Файл {new_name} уже существует, пропускаем {old_name}")
            continue

        os.rename(old_path, new_path)
        print(f"{old_name} -> {new_name}")

    print(f"Переименовано {len(images)} файлов.")


# Пример использования:
rename_images(r"dataset_temp", start_number=10001)