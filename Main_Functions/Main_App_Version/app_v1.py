import time
from datetime import datetime
from tkinter import *
from tkinter import font, filedialog

import cv2
import numpy as np
import supervision
import PIL.ImageTk, PIL.Image
from ultralytics import YOLO
from pyzbar.pyzbar import decode

from RealESRGAN import RealESRGAN

from Main_Functions.config_app import *

# ================================================== INIT DEFAULT SETTINGS ================================================== #
if "Init_WINDOW":
    window = Tk()

    SCREEN_MONITOR_WIDTH = window.winfo_screenwidth()
    SCREEN_MONITOR_HEIGHT = window.winfo_screenheight()

    WINDOW_POSITION_RIGHT = (SCREEN_MONITOR_WIDTH // 2) - (WINDOW_WIDTH // 2)
    WINDOW_POSITION_DOWN = (SCREEN_MONITOR_HEIGHT // 2) - (WINDOW_HEIGHT // 2)

    # Configure Width, Height, Position .geometry("window_width x window_height + window_position_right + window_position_down")
    window.geometry("{}x{}+{}+{}".format(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_POSITION_RIGHT, WINDOW_POSITION_DOWN))

    # Title
    window.title(TITLE_WINDOW)

    # Icon
    window.iconbitmap(ICON_WINDOW)

    # Background
    window["background"] = COLOR_BACKGROUND

    # Resizable
    window.resizable(False, False)

    # Parameter to handle realtime video - DO NOT CHANGE IT
    CHECK_MODE_CAMERA = False
    CAPTURE_FRAME = None

    # Initialize YOLOv8 model
    MODEL_YOLO_V8 = YOLO(PATH_MODEL_YOLO_V8)
    print("MODEL_YOLO_V8 successfully applied...")

    # Initialize Super-Resolution models
    MODEL_REAL_ESRGAN = RealESRGAN(DEVICE, scale=4)
    MODEL_REAL_ESRGAN.load_weights(PATH_MODEL_REAL_ESRGAN, download=False)
    print("MODEL_REAL_ESRGAN successfully applied...")


# ==================================================DEFINE COMPONENTS====================================================== #


class MenuBar:
    def __init__(self):
        self.menubar_ = Menu(window)
        window.config(menu=self.menubar_)

        self.File_()
        self.Settings_()
        self.Config_()
        self.Help_()
        self.About_()

    def File_(self):
        file_items = Menu(self.menubar_, tearoff=False)
        file_items.add_command(label="Open Image/Video...", command=ask_open_file)
        file_items.add_command(label="Open Webcam...", command=start_camera)
        file_items.add_separator()
        file_items.add_command(label="Exit", command=system_exit_)
        self.menubar_.add_cascade(label="File", menu=file_items, underline=False)

    def Settings_(self):
        setting_items = Menu(self.menubar_, tearoff=False)
        setting_items.add_command(label="Choose AI Model")
        setting_items.add_command(label="Setup Camera")
        self.menubar_.add_cascade(label="Settings", menu=setting_items, underline=False)

    def Config_(self):
        config_items = Menu(self.menubar_, tearoff=False)
        config_items.add_command(label="COM port")
        config_items.add_command(label="Format SFIS")
        self.menubar_.add_cascade(label="Config", menu=config_items, underline=False)

    def Help_(self):
        help_items = Menu(self.menubar_, tearoff=False)
        help_items.add_command(label="Welcome")
        help_items.add_command(label="Document")
        self.menubar_.add_cascade(label="Help", menu=help_items, underline=False)

    def About_(self):
        about_items = Menu(self.menubar_, tearoff=False)
        about_items.add_command(label="Version")
        about_items.add_command(label="Development Teams")
        self.menubar_.add_cascade(label="About", menu=about_items, underline=False)


if "Widgets":
    FONT_TEXT = font.Font(window, family="Calibri", size=18, weight="bold")

    frame0 = Frame(window, width=45, height=210, bg=window["background"])
    frame1 = Frame(window, width=220, height=220, bg=window["background"])

    pass_fail_quantity = Label(frame0, text="OK: 0\nNG: 0", font=FONT_TEXT, width=15, bg=window["background"])

    display_info_barcodes = Label(frame0, text=RETURN_RESULT["origin"], font=FONT_TEXT, width=15, height=5, bg=COLOR_ICE_BLUE)

    display_time = Label(frame0, text="", font=FONT_TEXT, width=15, bg=window["background"])

    display_img0 = Label(frame1, image="", width=WIDTH_HEIGHT_DISPLAY[0], height=WIDTH_HEIGHT_DISPLAY[1])


class App:
    def __init__(self):
        self.window = window

        self.frame0 = frame0
        self.frame0.grid(row=0, column=0, padx=0, pady=0, sticky="nesw")

        self.frame1 = frame1
        self.frame1.grid(row=0, column=1, padx=0, pady=0, sticky="nesw")

        # "DISPLAY_PASS_FAIL":
        self.display_pass_fail = display_info_barcodes
        self.display_pass_fail.grid(row=2, column=0, padx=5, pady=5, sticky="nesw")

        # "DISPLAY_TIME":
        self.display_time = display_time
        self.display_time.grid(row=3, column=0, padx=5, pady=5, sticky="nesw")

        # "PASS_FAIL_QUANTITY":
        self.pass_fail_quantity = pass_fail_quantity
        self.pass_fail_quantity.grid(row=4, column=0, padx=5, pady=5, sticky="nesw")

        # "START_BUTTON":
        self.start_btn = Button(frame0, text="START", font=FONT_TEXT, width=15, bg=COLOR_GREEN, command=start_camera)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="nesw")

        # "STOP_BUTTON":
        self.stop_btn = Button(frame0, text="STOP", font=FONT_TEXT, width=15, bg=COLOR_RED, command=stop_camera)
        self.stop_btn.grid(row=1, column=0, padx=5, pady=5, sticky="nesw")

        # "DISPLAY_CAMERA":
        self.display_img0 = display_img0
        self.display_img0.grid(row=0, column=0, padx=5, pady=5, sticky="nesw")


# ==================================================PROCESS FUNCTION====================================================== #

def start_camera():
    global CHECK_MODE_CAMERA, CAPTURE_FRAME
    stop_camera_opening()
    CHECK_MODE_CAMERA = True
    CAPTURE_FRAME = cv2.VideoCapture(VIDEO_SOURCE)  # Capture video frames, 0 is your default video camera
    print("[INFO] Opened camera...")
    update_main_display()


def stop_camera():
    """Destroy the root object and release all resources."""
    stop_camera_opening()
    print("[INFO] Closed camera...")


def stop_camera_opening():
    global CHECK_MODE_CAMERA
    CHECK_MODE_CAMERA = False
    if CAPTURE_FRAME:
        CAPTURE_FRAME.release()  # Release the camera
        display_img0.configure(image="")


def update_main_display():
    """Open the camera and update the frame."""
    if CHECK_MODE_CAMERA:
        ret, frame = CAPTURE_FRAME.read()  # Read frame from video stream
        if ret:  # Frame captured without any errors
            yolov8_results = get_result_yolov8(resize_width_height_yolov8(frame))  # Trả về ảnh barcode đã được cắt theo tọa độ để super-resolution nó

            display_main_image_result_from_yolov8(image=yolov8_results[0])
            enhance_super_resolution_barcode(images_=yolov8_results[1])
        display_img0.after(DELAY_UPDATE_FRAME, update_main_display)  # Call the function after "DELAY_UPDATE_FRAME" milliseconds


def enhance_super_resolution_barcode(images_):
    """
    Implement ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
    Link: https://paperswithcode.com/paper/esrgan-enhanced-super-resolution-generative
    :param images_:
    :return:
    """

    # TODO: triển khai các pretrained model của các
    #  phương pháp cũ lên đối tượng là ảnh barcode.

    # TODO:
    #  1. Đọc lại bài báo real-esrgan để hiểu rõ hơn phương pháp của mình
    #  2. Đọc các bài báo cũ để triển khai thành code
    #  3. Nhanh lên thời gian có hạn

    for i in range(len(images_)):
        start_time = time.time()

        # image = images_[i] * 1.0 / 255
        # image = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()
        # image_LR = image.unsqueeze(0)
        # image_LR = image_LR.to(DEVICE)

        # with torch.no_grad():
        #     time.sleep(0.05)
        #     output = MODEL_ESRGAN(image_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        image_pil = IMAGE_PIL.fromarray(images_[i]).convert('RGB')
        sr_image = MODEL_REAL_ESRGAN.predict(image_pil)

        sr_image = np.array(sr_image)

        output = Read_Barcode_QRCode(sr_image)

        file_name = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3] + ".jpg"
        cv2.imwrite(PATH_SAVE_INPUT_FILES + file_name, images_[i])
        cv2.imwrite(PATH_SAVE_OUTPUT_FILES + file_name, output)
        end_time = time.time()
        print("The time of Enhance Super Resolution Image is:", (end_time - start_time) * 10 ** 3, "ms")


def Read_Barcode_QRCode(image):
    for d in decode(image):
        cv2.rectangle(image, (d.rect.left, d.rect.top), (d.rect.left + d.rect.width, d.rect.top + d.rect.height), (255, 0, 0), 2)
        cv2.polylines(image, [np.array(d.polygon)], True, (0, 255, 0), 2)
        cv2.putText(image, d.data.decode(), (d.rect.left, d.rect.top + d.rect.height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image


def display_main_image_result_from_yolov8(image):
    frame_converted = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8))  # Convert image for PIL
    main_frame = PIL.ImageTk.PhotoImage(frame_converted)  # Convert image for tkinter

    display_img0.image = main_frame  # Hiển thị ảnh chính ra màn hình chính
    display_img0.configure(image=main_frame)  # Hiển thị ảnh chính ra màn hình chính


def ask_open_file():
    stop_camera_opening()
    window.filename = filedialog.askopenfile(
        initialdir=INITIAL_DIR,
        title="Select A File",
        filetypes=[("all files", "*.*"), ("jpg files", "*.jpg"), ("png files", "*.png"), ("mp4 files", "*.mp4")])
    # try:
    path = window.filename.name
    image = cv2.imread(path)
    handle_image(image)
    # except Exception as e:
    #     print("[INFO] Error: " + f"{e}" + " ===> " + "Please choose file type (.jpg, .png, .mp4)")


def handle_image(image_param):
    img_wh_yolov8 = resize_width_height_yolov8(image_param)
    result_yolov8 = get_result_yolov8(img_wh_yolov8)

    enhance_super_resolution_barcode(images_=result_yolov8[1])
    update_img0(result_yolov8[0])


def resize_width_height_yolov8(image):
    return cv2.resize(image, YOLOv8_IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA)


def update_img0(image_param):
    img_wh_display = cv2.resize(image_param, WIDTH_HEIGHT_DISPLAY, interpolation=cv2.INTER_AREA)
    img_converted = PIL.Image.fromarray(cv2.cvtColor(img_wh_display, cv2.COLOR_BGR2RGB))

    img = PIL.ImageTk.PhotoImage(img_converted)
    display_img0.image = img
    display_img0.configure(image=img)


def get_result_yolov8(image):
    image_copy = image.copy()

    box_annotator = supervision.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5,
        text_padding=2)
    result = MODEL_YOLO_V8(image)[0]
    detections_ = supervision.Detections.from_yolov8(result)

    lst_crop_img_barcodes = []
    for num_ in detections_.xyxy:
        x1, y1, x2, y2 = num_
        image_crop = image_copy[int(y1) - 2:int(y2) + 2, int(x1) - 2:int(x2) + 2]
        lst_crop_img_barcodes.append(image_crop)

    labels = [f"{LABELS[d[3]]} {d[2]:.2f}"
              for d
              in list(detections_)]
    ret = box_annotator.annotate(
        scene=image,
        detections=detections_,
        labels=labels
    )

    return ret, lst_crop_img_barcodes


def update_present_time_():
    my_time = datetime.now().strftime("%m/%d/%Y\n%I:%M:%S %p")
    display_time.config(text=my_time)
    display_time.after(1000, update_present_time_)


def system_exit_():
    stop_camera_opening()
    window.destroy()


def main():
    global window
    MenuBar()
    App()
    window.bind("<Control-o>", lambda x: ask_open_file())
    window.mainloop()
