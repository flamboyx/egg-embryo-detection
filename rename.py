import os


def rename_images(folder_path, start_number=10001):
    files = os.listdir(folder_path)
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in files if f.lower().endswith(extensions)]
    images.sort()

    for idx, old_name in enumerate(images):
        ext = os.path.splitext(old_name)[1]
        new_name = f"{start_number + idx}{ext}"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        if os.path.exists(new_path):
            print(f"Файл {new_name} уже существует, пропускаем {old_name}")
            continue

        os.rename(old_path, new_path)
        print(f"{old_name} -> {new_name}")

    print(f"Переименовано {len(images)} файлов.")


rename_images(r"dataset_temp", start_number=10001)