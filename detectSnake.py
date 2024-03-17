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
    """
    ค้นหาชื่อโฟลเดอร์ถัดไปเพื่อไม่ให้ซ้ำกับโฟลเดอร์ที่มีอยู่แล้ว
    ตัวอย่าง: 'runs/detect/exp', 'runs/detect/exp1', 'runs/detect/exp2', ...
    """
    counter = 0  # เริ่มต้นจาก 'exp' และหลังจากนั้นเป็น 'exp1', 'exp2', ...
    while True:
        if counter == 0:
            # สร้างชื่อโฟลเดอร์เริ่มต้นเป็น 'exp' ถ้าเป็นรอบแรก
            new_path = os.path.join(path_base, "exp")
        else:
            # ถ้าไม่ใช่รอบแรก, เพิ่มตัวเลขท้ายชื่อโฟลเดอร์ ('exp1', 'exp2', ...)
            new_path = os.path.join(path_base, f"exp{counter}")
        # ตรวจสอบว่าโฟลเดอร์ที่กำหนดมีอยู่แล้วหรือไม่
        if not os.path.exists(new_path):
            # ถ้าไม่มี, สร้างโฟลเดอร์และคืนค่าเส้นทางโฟลเดอร์ที่สร้างใหม่
            os.makedirs(new_path, exist_ok=True)
            return new_path
        # ถ้ามีแล้ว, เพิ่มตัวนับและวนลูปต่อ
        counter += 1


# ใช้ฟังก์ชันเพื่อรับชื่อโฟลเดอร์สำหรับบันทึกผลลัพธ์
# ฟังก์ชัน find_next_exp_dir จะค้นหาและสร้างชื่อโฟลเดอร์ใหม่ในรูปแบบที่ไม่ซ้ำกับโฟลเดอร์ที่มีอยู่แล้ว
# เพื่อป้องกันการเขียนทับข้อมูลเมื่อทำการทดลองหลายครั้ง
save_path_base = find_next_exp_dir()

# แสดงเส้นทางของโฟลเดอร์ที่ผลลัพธ์จะถูกบันทึก
# ช่วยให้ผู้ใช้ทราบว่าผลลัพธ์จากการทดลองครั้งนี้จะถูกเก็บไว้ที่ไหน
print(f"Results will be saved in: {save_path_base}")


def draw_labels_and_boxes(im0s, det, names, font_size=0.5, font_thickness=1):
    # วนลูปเพื่อดึงข้อมูลจากการตรวจจับที่ได้ โดยแต่ละการตรวจจับจะมีพิกัดขอบเขต (xyxy), ความมั่นใจ (conf), และคลาส (cls)
    for *xyxy, conf, cls in det:
        # สร้าง label จากชื่อคลาสและความมั่นใจที่ได้ และแปลง cls จาก float เป็น int เพื่อใช้เป็นดัชนี
        label = f"{names[int(cls)]} {conf:.2f}"
        # กำหนดจุดเริ่มต้น (c1) และจุดสิ้นสุด (c2) ของกรอบสี่เหลี่ยมจากพิกัด xyxy
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        # กำหนดสีของกรอบและข้อความเป็นสีเขียว
        color = (0, 255, 0)  # Green
        # วาดกรอบสี่เหลี่ยมรอบๆ วัตถุที่ตรวจจับได้
        cv2.rectangle(im0s, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
        # คำนวณขนาดของข้อความเพื่อใช้ในการวาดกรอบข้อความ
        t_size = cv2.getTextSize(
            label, 0, fontScale=font_size, thickness=font_thickness
        )[0]
        # ปรับขนาดและตำแหน่งของกรอบข้อความให้พอดีกับข้อความ
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # วาดกรอบข้อความสีเขียวเพื่อใส่ข้อความ label
        cv2.rectangle(im0s, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # วาดข้อความ label ลงบนกรอบ
        cv2.putText(
            im0s,
            label,
            (c1[0], c1[1] - 2),
            0,
            font_size,
            [225, 255, 255],
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )


def save_detected_images(
    im0s, det_binary, det_multiclass, path, names_binary, names_multiclass
):
    print(f"Saving detected images for {path}")
    save_path = str(Path("runs/detect/exp") / Path(path).name)
    annotator = Annotator(im0s, line_width=3, example=str(names_binary))

    # Process binary detections
    if len(det_binary):
        print("Binary Detections:")
        for *xyxy, conf, _ in det_binary:
            label = f"Snake {conf:.2f}"
            print(f"  {label}")
            annotator.box_label(
                xyxy, label, color=(0, 0, 255)
            )  # Red for binary detection

    # Process multiclass detections
    if len(det_multiclass):
        print("Multiclass Detections:")
        for *xyxy, conf, cls in det_multiclass:
            c = int(cls)
            label = f"{names_multiclass[c]} {conf:.2f}"
            print(f"  {label}")
            annotator.box_label(
                xyxy, label, color=(255, 0, 0)
            )  # Blue for multiclass detection

    cv2.imwrite(save_path, annotator.result())
    print(f"Image saved to {save_path}")


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
    # แปลง 'opt' จาก Namespace (ที่ได้จาก argparse) เป็น dictionary
    # เพื่อให้สามารถจัดการกับข้อมูลได้ง่ายขึ้น
    opt_dict = vars(opt)

    # ลบ 'conf_thres' และ 'iou_thres' ออกจาก dictionary ถ้ามี
    # เนื่องจากอาจไม่จำเป็นต้องใช้ในฟังก์ชัน run()
    # การลบออกจะป้องกันไม่ให้เกิดข้อผิดพลาดจากการส่งอาร์กิวเมนต์ที่ไม่ต้องการไปยังฟังก์ชัน
    opt_dict.pop("conf_thres", None)
    opt_dict.pop("iou_thres", None)

    # เรียกใช้ฟังก์ชัน run() พร้อมส่งข้อมูลการตั้งค่าที่ได้ปรับแต่งแล้ว
    # ใช้ **opt_dict เพื่อแยก dictionary ออกเป็นอาร์กิวเมนต์ที่ run() สามารถรับได้
    run(**opt_dict)


if __name__ == "__main__":
    # ส่วนนี้จะถูกเรียกใช้งานเมื่อไฟล์ถูกเรียกใช้เป็นสคริปต์หลัก
    # เรียกใช้ฟังก์ชัน parse_opt() เพื่อรับอาร์กิวเมนต์จาก command line
    opt = parse_opt()
    # เรียกใช้ฟังก์ชัน main() พร้อมส่งอาร์กิวเมนต์ที่ได้จาก parse_opt()
    main(opt)


# คำสั่งสำหรับการทดลอง
# ใช้คำสั่งด้านล่างเพื่อทดลองการตรวจจับจากโมเดล binary และ multiclass บนภาพที่อยู่ในโฟลเดอร์ 'test'
# python detectSnake.py
