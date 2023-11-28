import time
import datetime
import cv2
import numpy as np
import supervision
from ultralytics import YOLO
from pyzbar.pyzbar import decode
from RealESRGAN import RealESRGAN
from archs import RRDBNet_arch
from scripts.config import *


def load_model_yolov8(path_=PATH_MODEL_YOLO_V8):
    yolov8 = YOLO(path_)
    print(f"[INFO] YOLO_V8 loaded successfully, from {path_}")
    return yolov8


def load_model_realesrgan(
        path_=PATH_MODEL_REAL_ESRGAN,
        device=DEVICE,
        scale=4,
):
    realesrgan = RealESRGAN(device, scale)
    print(f"[INFO] REAL_ESRGAN loaded successfully, from {path_}")
    return realesrgan.load_weights(path_, download=False)


if __name__ == "__main__":
    # Load models
    MODEL_YOLOV8 = load_model_yolov8()
    MODEL_REALESRGAN = load_model_realesrgan()

    # Read image from video
    path = "D:/Projects/Datasets/ALBY_Datasets/Foxconn_LBE5A_dataset/Videos/LB5AG2SD.US_video.avi"
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (600, 600), interpolation=cv2.INTER_AREA)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    print("[Debug...]")
