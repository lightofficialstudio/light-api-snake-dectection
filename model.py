import torch
import os


def get_yolov5():
    # ใช้ os.path.abspath เพื่อแปลงเป็น absolute path
    model_binary_path = os.path.abspath(os.path.join("model", "model_binary.pt"))
    model_multiclass_path = os.path.abspath(
        os.path.join("model", "model_multiclass.pt")
    )

    model_binary = torch.hub.load(
        "yolo",  # ตรวจสอบว่า "yolo" นี้อ้างอิงถึง path หรือ repo ที่ถูกต้องหรือไม่
        "custom",
        path=model_binary_path,
        source="local",
    )
    model_multiclass = torch.hub.load(
        "yolo",  # เช่นเดียวกับด้านบน
        "custom",
        path=model_multiclass_path,
        source="local",
    )

    model_multiclass.conf = 0.01  # ตั้งค่า confidence

    return model_multiclass  # Return model_multiclass โดยไม่ต้องตรวจสอบ conf
