import PIL.Image
import torch

IMAGE_PIL = PIL.Image

TITLE_WINDOW = "AICT-LAB-702 - National Kaohsiung University Science and Technology"
ICON_WINDOW = "E:/Study/SDDLE/Icons/icon_nkust.ico"

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
PATH_MODEL_YOLO_V8 = "E:/Study/SDDLE/Model_YOLOv8/09_07_2023_08h59m/weights/best.pt"

# ESRGAN model path
PATH_MODEL_REAL_ESRGAN = "E:/Study/SDDLE/Model_REAL_ESRGAN/pre_trained_model/weights/RealESRGAN_x4.pth"

# DEVICE GPU or CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Webcam
VIDEO_SOURCE = 0

# Delay update frame
DELAY_UPDATE_FRAME = 1

# Save dir-in, dir-out
PATH_SAVE_INPUT_FILES = "E:/Study/SDDLE/Save_Intput" + "/"
PATH_SAVE_OUTPUT_FILES = "E:/Study/SDDLE/Save_Output" + "/"


"""
Tính năng của chương trình:
1. Model được training on 8000 ảnh barcodes, cỡ 416x416.
2. Khả năng đọc file: ask_open_file()
3. khả năng kết nối webcam đọc video. 
4. Đọc ảnh -> resize ảnh -> đưa vào model yolov8 -> hiển thị kết quả ảnh.
5. Hiển đồng hồ thời gian thực 
"""