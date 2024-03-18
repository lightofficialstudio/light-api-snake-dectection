import json
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device


def detect_snake_image(
    image_stream,
    weights_binary="model_binary.pt",
    weights_multiclass="model_multiclass.pt",
    imgsz=224,
    device="",
):
    ROOT = (
        Path(__file__).resolve().parent
    )  # Use pathlib for cross-platform compatibility
    weights_binary = ROOT / weights_binary
    weights_multiclass = ROOT / weights_multiclass

    print(f"Using device: {device}")
    device = select_device(device)
    imgsz = check_img_size(imgsz, s=32)
    print(f"Image size: {imgsz}")

    model_binary = DetectMultiBackend(
        weights_binary.as_posix(), device=device, dnn=False
    )
    model_multiclass = DetectMultiBackend(
        weights_multiclass.as_posix(), device=device, dnn=False
    )

    print("Models loaded successfully.")

    stride_binary = (
        torch.tensor(model_binary.stride).max()
        if not isinstance(model_binary.stride, int)
        else model_binary.stride
    )
    stride_multiclass = (
        torch.tensor(model_multiclass.stride).max()
        if not isinstance(model_multiclass.stride, int)
        else model_multiclass.stride
    )
    stride = max(stride_binary, stride_multiclass)
    print(f"Max stride: {stride}")

    dataset = LoadImages([image_stream], img_size=imgsz, stride=stride, auto=True)
    snake_details = []
    for data in dataset:
        path, img, im0s, _ = data[:4]
        img = torch.from_numpy(img).to(device)
        img = img.half() if model_binary.fp16 else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred_binary = model_binary(img, augment=False)[0]
        pred_multiclass = model_multiclass(img, augment=False)[0]
        print(f"Detections: {len(pred_multiclass)}")

        pred_binary = non_max_suppression(
            pred_binary, 0.4, 0.5, classes=None, agnostic=False
        )
        pred_multiclass = non_max_suppression(
            pred_multiclass, 0.05, 0.01, classes=None, agnostic=False
        )

        for i, det in enumerate(pred_multiclass):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()
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
                    )
                    snake_details.append(
                        {
                            "class": cls.item(),
                            "class_name": (
                                model_multiclass.names[cls.item()]
                                if model_multiclass.names
                                else ""
                            ),
                            "confidence": conf.item(),
                            "probability": conf.item() * 100,
                        }
                    )
                    print(f"Detected: {snake_details[-1]}")

    return json.dumps(snake_details, indent=2)
