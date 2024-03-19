import torch


def get_yolov5():
    model_binary = torch.hub.load(
        "./yolov5", "custom", path="./model/model_binary.pt", source="local"
    )
    model_multiclass = torch.hub.load(
        "./yolov5", "custom", path="./model/model_multiclass.pt", source="local"
    )

    # ตั้งค่า conf สำหรับ model_multiclass เป็น confident_value ไม่ว่าเงื่อนไขจะเป็นอย่างไร
    model_multiclass.conf = 0.01

    if model_binary.conf > 0.4:
        return model_multiclass
    else:
        # ตัวอย่าง: หากคุณต้องการ return model_multiclass ไม่ว่าอย่างไร
        # หรือคุณอาจจะต้องการ handle การไม่เข้าเงื่อนไขในวิธีอื่น
        return model_multiclass
