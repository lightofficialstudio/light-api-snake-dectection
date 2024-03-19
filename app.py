from flask import Flask, request, jsonify, Blueprint
from model import get_yolov5
from PIL import Image
from io import BytesIO
import base64

model_snake = get_yolov5()

app = Flask(__name__)  # สร้าง Flask application

# สร้าง Blueprint เพื่อเก็บ route ทั้งหมด
main = Blueprint("main", __name__)


@main.route("/", methods=["GET"])
def index():
    print("User is on the index page")
    return "Welcome to Snake Detection API"


@main.route("/predict", methods=["POST"])
def detect_image():
    print("User is on the predict page")
    if model_snake is None:
        return jsonify({"error": "Model not loaded properly"}), 500

    image = request.files["image"]
    img = Image.open(BytesIO(image.read()))
    results = model_snake(img, size=244)
    results_data = results.pandas().xyxy[0]

    label_result = []
    for _, row in results_data.iterrows():
        label_result.append(
            {
                "class_name": row["name"],
                "confidence": row["confidence"],
            }
        )

    return jsonify({"labels": label_result})


# เพิ่ม Blueprint ลงใน app
app.register_blueprint(main)

if __name__ == "__main__":
    print("Hello, World!")
    app.run(host="0.0.0.0", port=8080, debug=False)
