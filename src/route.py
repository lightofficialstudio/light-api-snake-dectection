from flask import Flask, request, jsonify, Blueprint
from model import get_yolov5
from PIL import Image
from io import BytesIO
import base64

model_snake = get_yolov5()

app = Blueprint("main", __name__)


@app.route("/", methods=["GET"])
def index():
    print("User is on the index page")
    return "Welcome to Snake Detection API"


@app.route("/predict", methods=["POST"])
def detect_image():
    print("User is on the predict page")
    if model_snake is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    image = request.files["image"]
    img = Image.open(BytesIO(image.read()))
    results = model_snake(img, size=244)
    results_data = results.pandas().xyxy[0]  # เข้าถึงผลลัพธ์เต็มรูปแบบ

    label_result = []
    for _, row in results_data.iterrows():
        label_result.append(
            {
                "class_name": row["name"],  # ตัวอย่างการเข้าถึง class_name
                "confidence": row["confidence"],
                # คุณสามารถเพิ่ม fields อื่นๆ ที่ต้องการได้
            }
        )

    return jsonify({"labels": label_result})
