import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2

# Initialize and add YOLOv5 root directory to sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors


def find_next_exp_dir(path_base="runs/detect"):
    counter = 0
    while True:
        new_path = os.path.join(path_base, f"exp{counter}" if counter else "exp")
        if not os.path.exists(new_path):
            os.makedirs(new_path, exist_ok=True)
            return new_path
        counter += 1


save_path_base = find_next_exp_dir()
print(f"Results will be saved in: {save_path_base}")


def draw_labels_and_boxes(im0s, det, names, font_size=0.5, font_thickness=1):
    for *xyxy, conf, cls in reversed(det):
        label = f"{names[int(cls)]} {conf:.2f}"
        Annotator(im0s, line_width=2).box_label(
            xyxy, label, color=colors(int(cls), True)
        )


def detect_snake_image(
    image_path,
    weights_binary="model_binary.pt",
    weights_multiclass="model_multiclass.pt",
    imgsz=224,
    device="",
):
    device = select_device(device)
    imgsz = check_img_size(imgsz)

    # Correctly handle paths
    weights_binary = str(ROOT / weights_binary)
    weights_multiclass = str(ROOT / weights_multiclass)

    model_binary = DetectMultiBackend(weights_binary, device=device, dnn=False)
    model_multiclass = DetectMultiBackend(weights_multiclass, device=device, dnn=False)

    stride = max(int(model_binary.stride.max()), int(model_multiclass.stride.max()))
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=True)

    results = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if model_binary.fp16 else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred_binary = model_binary(img, augment=False)[0]
        pred_multiclass = model_multiclass(img, augment=False)[0]

        # Apply NMS
        pred_binary = non_max_suppression(
            pred_binary, 0.4, 0.5, classes=None, agnostic=False
        )
        pred_multiclass = non_max_suppression(
            pred_multiclass, 0.25, 0.5, classes=None, agnostic=False
        )

        for i, det in enumerate(pred_multiclass):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    xywh = (
                        (
                            xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                            / torch.tensor(
                                [
                                    im0s.shape[1],
                                    im0s.shape[0],
                                    im0s.shape[1],
                                    im0s.shape[0],
                                ]
                            )
                        )
                        .view(-1)
                        .tolist()
                    )  # normalized xywh
                    line = (cls, *xywh, conf)  # label format
                    results.append(line)

    return results


@smart_inference_mode()
def run(weights_binary, weights_multiclass, source, imgsz=224, device=""):
    # กำหนดค่าความมั่นใจสำหรับการตรวจจับโมเดล binary และ multiclass
    binary_conf_thres = 0.5  # กำหนดค่าความมั่นใจขั้นต่ำที่ 50% สำหรับโมเดล binary
    multiclass_conf_thres = 0.05  # กำหนดค่าความมั่นใจขั้นต่ำที่ 5% สำหรับโมเดล multiclass
    iou_thres = 0.45  # กำหนดค่า IOU สำหรับกระบวนการ NMS (Non-Maximum Suppression)

    print("Initializing...")
    device = select_device(device)  # เลือกอุปกรณ์สำหรับการคำนวณ (CPU หรือ GPU)
    imgsz = check_img_size(imgsz)  # ตรวจสอบและปรับขนาดของภาพตามโมเดล

    # โหลดโมเดลสำหรับการตรวจจับ binary และ multiclass
    model_binary = DetectMultiBackend(weights_binary, device=device, dnn=False)
    names_binary = model_binary.names  # รับชื่อคลาสจากโมเดล binary
    model_multiclass = DetectMultiBackend(weights_multiclass, device=device, dnn=False)
    names_multiclass = model_multiclass.names  # รับชื่อคลาสจากโมเดล multiclass

    # โหลดข้อมูลสำหรับการตรวจจับ
    print(f"Loading data from {source}...")
    dataset = LoadImages(
        source,
        img_size=imgsz,
        stride=max(model_binary.stride, model_multiclass.stride),
        auto=True,
    )

    # สร้างไดเรกทอรีสำหรับบันทึกผลลัพธ์
    os.makedirs("runs/detect/exp", exist_ok=True)

    for path, img, im0s, _, _ in dataset:
        im0s_resized = cv2.resize(im0s, (224, 224))  # ปรับขนาดภาพต้นฉบับเป็น 224x224
        img = np.ascontiguousarray(
            im0s_resized.transpose(2, 0, 1)
        )  # ปรับรูปแบบของภาพเพื่อเข้ากับโมเดล
        img = (
            torch.from_numpy(img).to(device).float() / 255.0
        )  # ปรับข้อมูลภาพเป็น tensor และ normalize
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # เพิ่มมิติสำหรับ batch หากจำเป็น

        # ทำนายด้วยโมเดล binary
        pred_binary = model_binary(img, augment=False, visualize=False)
        pred_binary = non_max_suppression(
            pred_binary, binary_conf_thres, iou_thres, classes=None, agnostic=False
        )

        # ตรวจสอบและทำนายด้วยโมเดล multiclass หากมีการตรวจจับจากโมเดล binary
        pred_multiclass = []  # กำหนดลิสต์ว่างสำหรับผลลัพธ์จากโมเดล multiclass
        if len(pred_binary[0]):
            pred_multiclass = model_multiclass(img, augment=False, visualize=False)
            pred_multiclass = non_max_suppression(
                pred_multiclass,
                multiclass_conf_thres,
                iou_thres,
                classes=None,
                agnostic=False,
            )

            # บันทึกภาพที่มีการวาดกรอบและข้อความบ่งบอกการตรวจจับบนภาพ
            save_detected_images(
                im0s_resized,
                pred_binary[0],
                pred_multiclass[0] if len(pred_multiclass) else [],
                path,
                names_binary,
                names_multiclass,
            )


def parse_opt():
    # สร้างออบเจกต์สำหรับการจัดการอาร์กิวเมนต์ที่ส่งจาก command line
    parser = argparse.ArgumentParser()

    # เพิ่มอาร์กิวเมนต์สำหรับโมเดล binary (การตรวจจับว่าเป็นงูหรือไม่)
    # ผู้ใช้สามารถระบุไฟล์น้ำหนักของโมเดล binary ผ่าน command line
    parser.add_argument(
        "--weights-binary",
        type=str,
        default=ROOT
        / "model_binary.pt",  # ค่าเริ่มต้นคือไฟล์ 'model_binary.pt' ในโฟลเดอร์ ROOT
        help="path to binary model weights file",  # ข้อความช่วยเหลือสำหรับอาร์กิวเมนต์นี้
    )

    # เพิ่มอาร์กิวเมนต์สำหรับโมเดล multiclass (การตรวจจับประเภทของงู)
    # ผู้ใช้สามารถระบุไฟล์น้ำหนักของโมเดล multiclass ผ่าน command line
    parser.add_argument(
        "--weights-multiclass",
        type=str,
        default=ROOT
        / "model_multiclass.pt",  # ค่าเริ่มต้นคือไฟล์ 'model_multiclass.pt' ในโฟลเดอร์ ROOT
        help="path to multiclass model weights file",  # ข้อความช่วยเหลือสำหรับอาร์กิวเมนต์นี้
    )

    # เพิ่มอาร์กิวเมนต์สำหรับแหล่งข้อมูล (ภาพ/วิดีโอ) ที่จะใช้ในการตรวจจับ
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "test",  # ค่าเริ่มต้นคือโฟลเดอร์ 'test' ในโฟลเดอร์ ROOT
        help="source",  # ข้อความช่วยเหลือสำหรับอาร์กิวเมนต์นี้
    )

    # เพิ่มอาร์กิวเมนต์สำหรับขนาดภาพในการทำนาย (inference)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,  # ค่าเริ่มต้นคือ 224x224 pixels
        help="inference size (pixels)",  # ข้อความช่วยเหลือสำหรับอาร์กิวเมนต์นี้
    )

    # เพิ่มอาร์กิวเมนต์สำหรับการเลือกอุปกรณ์ในการคำนวณ (CPU หรือ GPU)
    parser.add_argument(
        "--device",
        default="",  # ค่าเริ่มต้นคืออุปกรณ์เริ่มต้น (อาจเป็น CPU หรือ GPU ตามการตั้งค่า)
        help="cuda device, i.e. 0 or 0,1,2,3 or cpu",  # ข้อความช่วยเหลือสำหรับอาร์กิวเมนต์นี้
    )

    # คืนค่าอาร์กิวเมนต์ที่ได้รับจาก command line กลับเป็นออบเจกต์
    return parser.parse_args()


def main(opt):
    print(f"Detecting snakes in images at {opt.source}")
    results = detect_snake_image(
        opt.source, opt.weights_binary, opt.weights_multiclass, opt.imgsz, opt.device
    )
    print(results)


# หากต้องการให้ฟังก์ชันนี้รองรับการถูกเรียกใช้งานโดยตรงจาก command line สามารถเพิ่มโค้ดด้านล่างนี้
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-binary", type=str, default="yolov5s.pt", help="binary model path"
    )
    parser.add_argument(
        "--weights-multiclass",
        type=str,
        default="yolov5s.pt",
        help="multiclass model path",
    )
    parser.add_argument(
        "--source", type=str, default="data/images", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--imgsz", type=int, default=224, help="inference size (pixels)"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    opt = parser.parse_args()

    main(opt)
