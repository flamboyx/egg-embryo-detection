from ultralytics import YOLO
import cv2
import torch


def main():
    model_path = 'runs/detect/egg_detector/yolov8n_eggs-4/weights/last.pt'

    video_path = 'test_video.mp4'
    output_path = 'output_video.mp4'

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, device='cuda')

        annotated = results[0].plot()

        out.write(annotated)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f'Обработано кадров: {frame_count}')

    cap.release()
    out.release()
    print(f'Готово. Результат сохранён в {output_path}')


if __name__ == '__main__':
    main()
