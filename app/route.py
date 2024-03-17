import secrets
from flask import request, jsonify, Blueprint
from werkzeug.utils import secure_filename
from flask import current_app as app
import os
from pathlib import Path

# การนำเข้าฟังก์ชัน detect_snake_image จากไฟล์ detectSnakeAPI.py
from detectSnakeAPI import detect_snake_image

main = Blueprint("main", __name__)


@main.route("/", methods=["GET"])
def hello():
    return jsonify({"message": "สวัสดี นี่มิลเอง!"})


@main.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # สร้างชื่อไฟล์ที่ปลอดภัยและไม่ซ้ำกัน
    original_filename = secure_filename(image_file.filename)
    random_hex = secrets.token_hex(8)
    _, file_ext = os.path.splitext(original_filename)
    unique_filename = f"{random_hex}{file_ext}"

    # สร้างเส้นทางสำหรับบันทึกไฟล์
    image_path = Path(app.config["UPLOAD_FOLDER"]) / unique_filename

    # บันทึกไฟล์
    image_file.save(image_path)

    # เรียกใช้ฟังก์ชัน detect_snake_image จาก detectSnakeAPI.py
    detection_results = detect_snake_image(str(image_path))

    return jsonify(detection_results)
