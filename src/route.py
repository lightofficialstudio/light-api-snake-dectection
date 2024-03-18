from flask import Flask, request, jsonify, Blueprint
from werkzeug.utils import secure_filename
import os
import io
import json
import tempfile
from pathlib import Path

from detect_snake import detect_snake_image

main = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@main.route("/", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, this is the snake detection API!"})


@main.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": "Unsupported file format"}), 400

    try:
        # Create a temporary file for image processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file_path = Path(
                temp_file.name
            )  # Using Path for cross-platform compatibility
            image_file.save(temp_file_path)  # Direct use of Path with .save()

            # Call the snake detection function
            print(f"Temporary file path: {temp_file_path}")
            detection_results = detect_snake_image(
                str(temp_file_path)
            )  # Convert Path to string when passing to the function

            detection_json = json.loads(detection_results)

            # Process and format snake details
            snake_details = [
                {
                    "class": prediction["class"],
                    "class_name": prediction.get("class_name", ""),
                    "confidence": prediction["confidence"],
                    "probability": prediction["confidence"] * 100,
                }
                for prediction in detection_json
            ]

            # Return successful response with snake details
            return jsonify(snake_details)

    except Exception as e:
        return (
            jsonify({"error": f"Failed in detect_snake_image function: {str(e)}"}),
            500,
        )

    finally:
        # Always delete the temporary file after processing (if created)
        if (
            temp_file_path.exists()
        ):  # Use .exists() from Path for checking file existence
            temp_file_path.unlink(
                missing_ok=True
            )  # Use unlink for file removal, compatible with pathlib


app = Flask(__name__)
app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)
