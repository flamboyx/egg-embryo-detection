import cv2
import os


def extract_frames(video_path, output_dir, frame_step=30):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            out_path = os.path.join(output_dir, f'frame_{saved:05d}.jpg')
            cv2.imwrite(out_path, frame)
            saved += 1
            print(f'Сохранено: {out_path}')
        count += 1
    cap.release()
    print(f'Извлечено {saved} кадров.')


extract_frames('test_video.mp4', 'frames_for_labeling', frame_step=15)
