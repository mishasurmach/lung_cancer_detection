# proxy_server.py
import os

import requests
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:8888/invocations")


@app.route("/", methods=["GET"])
def index():
    "Displays the main page."
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload, forwards it to an MLflow server, and renders the prediction
    result.

    This route processes a POST request from the HTML form that includes a file upload.
    It performs the following steps:

    1. Validates the presence of a file in the request.
    2. Saves the uploaded file to the configured UPLOAD_FOLDER.
    3. Constructs a JSON payload with the absolute file path using the
    'dataframe_records' format, which is compatible with MLflow PyFunc models.
    4. Sends the payload to the MLflow model's /invocations endpoint.
    5. Parses the JSON response from the MLflow server.
    6. Extracts the prediction result from the response.
    7. Renders the 'index.html' template with the prediction and image preview.

    Returns:
        str: Rendered HTML with prediction and uploaded image.
        tuple: Error message and HTTP code in case of failure.
    """
    if "file" not in request.files:
        return "No file in request", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file was chosen", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    payload = {"dataframe_records": [{"data": filepath}]}

    try:
        resp = requests.post(MLFLOW_URL, json=payload)
    except requests.exceptions.RequestException as e:
        return f"Error connecting to MLflow: {e}", 500

    if resp.status_code != 200:
        return f"MLflow error {resp.status_code}: {resp.text}", 500

    result = resp.json()

    if isinstance(result, dict) and "predictions" in result:
        preds = result["predictions"]
    else:
        preds = result

    try:
        prediction = preds[0]
    except (IndexError, TypeError):
        prediction = preds

    class_labels = {
        0: "Lung adenocarcinoma",
        1: "Lung benign tissue",
        2: "Lung squamous cell carcinoma",
    }

    prediction_label = class_labels.get(prediction, f"Unknown class ({prediction})")

    return render_template(
        "index.html",
        prediction=prediction_label,
        image_url=url_for("uploaded_file", filename=filename),
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serves uploaded image files from the uploads directory.

    This route enables HTML templates to display the uploaded image
    by generating URLs like /uploads/<filename>.

    Args:
        filename (str): Name of the file to serve.

    Returns:
        Response: Flask response object that serves the file.
    """
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
