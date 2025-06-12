import torch
from ultralytics import YOLO
import argparse

def train_yolo(model_size='yolov8s', data_path='dataset.yaml', epochs=50, imgsz=640, batch=16):
    model = YOLO(f'{model_size}.pt')  # Load a pre-trained YOLOv8 model
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/train',
        name=f'{model_size}_finetuned',
        exist_ok=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='yolov8s', help='YOLO model size (n, s, m, l, x)')
    parser.add_argument('--data_path', type=str, default='dataset.yaml', help='Path to dataset YAML')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')

    args = parser.parse_args()
    train_yolo(args.model_size, args.data_path, args.epochs, args.imgsz, args.batch)
