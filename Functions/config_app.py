import PIL.Image
import torch
import os

ROOT_PATH = os.getcwd() + "/"

INITIAL_DIR = ROOT_PATH + "Data/Labels_Factory"

IMAGE_PIL = PIL.Image

TITLE_WINDOW = "AICT-LAB-702 - National Kaohsiung University Science and Technology"
ICON_WINDOW = ROOT_PATH + "Icons/icon_nkust.ico"

# Kích thước cửa sổ của App
WINDOW_WIDTH = 650
WINDOW_HEIGHT = 450

COLOR_BLACK = "#000000"
COLOR_BROWN = "#726461"
COLOR_RED = "#ff3333"
COLOR_BLUE = "#53D4F7"
COLOR_GREEN = "#00b359"
COLOR_WHITE = "#ffffff"
COLOR_PURPLE = "#a64dff"
COLOR_YELLOW = "#ffff00"
COLOR_ICE_BLUE = "#99FFFF"
COLOR_RUSSIAN_VIOLET = "#32174D"
COLOR_CHINESE_WHITE = "#e2e5de"
COLOR_LOTION = "#fefdfa"
COLOR_DARK_YELLOW = "#F5F5DC"

COLOR_BACKGROUND = COLOR_LOTION

# Trả về kết quả OK hay NG
RETURN_RESULT = {
    "origin": "...",
    "OK": "OK",
    "NG": "NG"
}

# Tất cả các ảnh đầu vào đều được resize với kích thước "YOLOv8_IMAGE_INPUT_SIZE"
YOLOv8_IMAGE_INPUT_SIZE = [416, 416]

# Hiển thị video tại của sổ chính
WIDTH_HEIGHT_DISPLAY = [416, 416]

# Hiển thị video tại các chức năng check sau này
WH_SUB_DISPLAY = (300, 150)

# Label các đối tượng
LABELS = {
    0: "Barcode",
    1: "QR_code"
}

# YOLOv8 model path
PATH_MODEL_YOLO_V8 = ROOT_PATH + "Models/Model_YOLOv8/09_07_2023_08h59m/weights/best.pt"

# REAL-ESRGAN model path
PATH_MODEL_REAL_ESRGAN = ROOT_PATH + "Models/Model_REAL_ESRGAN/REAL_ESRGAN_models/RealESRGAN_x4.pth"

# ESR-GAN model path
PATH_MODEL_ESRGAN = ROOT_PATH + "Models/Model_REAL_ESRGAN/Previous_Works/ESRGAN/RRDB_ESRGAN_x4.pth"

# DAN model path
PATH_MODEL_DAN = ROOT_PATH + ""

# CDC model path
PATH_MODEL_CDC = ROOT_PATH + ""

# REAL-SR model path
PATH_MODEL_REALSR = ROOT_PATH + ""

# BSR-GAN model path
PATH_MODEL_BSRGAN = ROOT_PATH + ""

# AC-GAN model path
PATH_MODEL_ACGAN = ROOT_PATH + ""

# SR-GAN model path
PATH_MODEL_SRGAN = ROOT_PATH + ""

# DEVICE GPU or CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Webcam
VIDEO_SOURCE = 0

# Delay update frame
DELAY_UPDATE_FRAME = 1

# Save dir-in, dir-out
PATH_SAVE_INPUT_FILES = ROOT_PATH + "Reports/Save_Inputs" + "/"
PATH_SAVE_OUTPUT_FILES = ROOT_PATH + "Reports/Save_Outputs" + "/"


"""
Tính năng của chương trình:
1. Model được training on 8000 ảnh barcodes, cỡ 416x416.
2. Khả năng đọc file: ask_open_file()
3. khả năng kết nối webcam đọc video. 
4. Đọc ảnh -> resize ảnh -> đưa vào model yolov8 -> hiển thị kết quả ảnh.
5. Hiển đồng hồ thời gian thực 
"""