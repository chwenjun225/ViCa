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


def load_model_realesrgan(path_=PATH_MODEL_REAL_ESRGAN, device=DEVICE, scale=4):
    realesrgan = RealESRGAN(device, scale)
    print(f"[INFO] REAL_ESRGAN loaded successfully, from {path_}")
    return realesrgan.load_weights(path_, download=False)


def decode_barcode(image):
    for d in decode(image):
        cv2.rectangle(image, (d.rect.left, d.rect.top), (d.rect.left + d.rect.width, d.rect.top + d.rect.height), (255, 0, 0), 2)
        cv2.polylines(image, [np.array(d.polygon)], True, (0, 255, 0), 2)
        cv2.putText(image, d.data.decode(), (d.rect.left, d.rect.top + d.rect.height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image


def realesrgan_image_processing(images_):
    """
    The implementation of Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic data
    Paper link: https://arxiv.org/pdf/2107.10833v2.pdf
    :param images_: list[array]
    :return:
    """
    for i in range(len(images_)):
        start_time = time.time()
        image_pil = IMAGE_PIL.fromarray(images_[i]).convert('RGB')
        sr_img = MODEL_REALESRGAN.predict(image_pil)
        sr_img = np.array(sr_img)
        img_output = decode_barcode(sr_img)
        # file_name = "REAL_ESRGAN_" + datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3] + ".jpg"
        # cv2.imwrite(PATH_SAVE_INPUT_FILES + file_name, images_[i])
        # cv2.imwrite(PATH_SAVE_OUTPUT_FILES + file_name, img_output)
        end_time = time.time()
        print(f"--- REAL_ESRGAN time reference:", round((end_time - start_time) * 10 ** 3), "ms")
    print("\n")


def result_from_yolov8(image):
    image_copy = image.copy()
    box_annotator = supervision.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5, text_padding=2)
    result = MODEL_YOLOV8(image)[0]
    detections_ = supervision.Detections.from_yolov8(result)
    lst_crop_img_barcodes = []
    for num_ in detections_.xyxy:
        x1, y1, x2, y2 = num_
        image_crop = image_copy[int(y1) - 2:int(y2) + 2, int(x1) - 2:int(x2) + 2]
        lst_crop_img_barcodes.append(image_crop)
    labels = [f"{LABELS[d[3]]} {d[2]:.2f}"
              for d
              in list(detections_)]
    return (box_annotator.annotate(scene=image, detections=detections_, labels=labels),
            lst_crop_img_barcodes)


if __name__ == "__main__":

    MODEL_YOLOV8 = load_model_yolov8()
    MODEL_REALESRGAN = load_model_realesrgan()

    path = "D:/Projects/Datasets/ALBY_Datasets/Foxconn_LBE5A_dataset/Videos/LB5AG2SD.US_video.avi"
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            exit()

        ret_from_yolov8 = result_from_yolov8(image=frame)

        realesrgan_image_processing(images_=ret_from_yolov8[1])

        image_resize = cv2.resize(ret_from_yolov8[0], (1000, 800), interpolation=cv2.INTER_AREA)

        cv2.imshow('image_resize', image_resize)
        if cv2.waitKey(45) == ord('q'):

            break
    cap.release()
    cv2.destroyAllWindows()

    print("[Debug...]")
