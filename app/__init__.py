# In __init__.py

from flask import Flask
from app.route import main
import os


def create_app():
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = "uploads"

    # Ensure the upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    app.register_blueprint(main)
    return app
