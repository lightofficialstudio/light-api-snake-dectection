import torch
import os


def get_yolov5():
    yolo_path = os.path.join("yolo")
    model_binary_path = os.path.join("model", "model_binary.pt")
    model_multiclass_path = os.path.join("model", "model_multiclass.pt")

    model_binary = torch.hub.load(
        yolo_path,
        "custom",
        path=model_binary_path,
        source="local",
    )
    model_multiclass = torch.hub.load(
        yolo_path,
        "custom",
        path=model_multiclass_path,
        source="local",
    )

    # ตั้งค่า conf สำหรับ model_multiclass เป็น confident_value ไม่ว่าเงื่อนไขจะเป็นอย่างไร
    model_multiclass.conf = 0.01

    if model_binary.conf > 0.4:
        return model_multiclass
    else:
        return model_multiclass
