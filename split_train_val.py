import os
import random
import shutil

new_src_dir = 'dataset_temp'
dataset_root = 'eggs_dataset'


def move_pair(img_name, dest_img_dir, dest_lbl_dir):
    base = os.path.splitext(img_name)[0]
    txt_name = base + '.txt'
    src_img = os.path.join(new_src_dir, img_name)
    src_txt = os.path.join(new_src_dir, txt_name)
    dst_img = os.path.join(dest_img_dir, img_name)
    dst_txt = os.path.join(dest_lbl_dir, txt_name)

    if not os.path.exists(src_txt):
        print(f'Предупреждение: {txt_name} не найден для {img_name}, пропускаем')
        return False

    if not os.path.exists(dst_img):
        shutil.move(src_img, dst_img)
    else:
        print(f'Файл {img_name} уже существует в {dest_img_dir}, пропускаем')
        return False

    if not os.path.exists(dst_txt):
        shutil.move(src_txt, dst_txt)
    else:
        print(f'Файл {txt_name} уже существует в {dest_lbl_dir}, пропускаем')
        return False
    return True


def main():
    train_img_dir = os.path.join(dataset_root, 'images', 'train')
    train_lbl_dir = os.path.join(dataset_root, 'labels', 'train')
    val_img_dir = os.path.join(dataset_root, 'images', 'val')
    val_lbl_dir = os.path.join(dataset_root, 'labels', 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    new_images = [f for f in os.listdir(new_src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f'Найдено новых изображений: {len(new_images)}')

    random.seed(42)
    random.shuffle(new_images)

    val_ratio = 0.2
    val_count = int(len(new_images) * val_ratio)
    val_images_new = new_images[:val_count]
    train_images_new = new_images[val_count:]

    print(f'В train попадает {len(train_images_new)}, в val {len(val_images_new)}')

    copied_train = 0
    skipped_train = 0
    for img in train_images_new:
        if move_pair(img, train_img_dir, train_lbl_dir):
            copied_train += 1
        else:
            skipped_train += 1

    copied_val = 0
    skipped_val = 0
    for img in val_images_new:
        if move_pair(img, val_img_dir, val_lbl_dir):
            copied_val += 1
        else:
            skipped_val += 1

    print(f'\nTrain: перемещено {copied_train}, пропущено {skipped_train}')
    print(f'Val: перемещено {copied_val}, пропущено {skipped_val}')
    print(f'Всего пропущено пар: {skipped_train + skipped_val}')

    for f in os.listdir(new_src_dir):
        file_path = os.path.join(new_src_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print('\nИтоговое количество изображений:')
    print(f'Train: {len(os.listdir(train_img_dir))}')
    print(f'Val: {len(os.listdir(val_img_dir))}')


if __name__ == '__main__':
    main()
