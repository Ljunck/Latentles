from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid

from processor import encode_new_image, decode_embedding
from encoder import create_embeddings

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "data/output"
EMBEDDING_FOLDER = "data/embeddings"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Automatically train PCA if the saved PCA files do not exist
if not os.path.exists(os.path.join(EMBEDDING_FOLDER, "mean.npy")):
    print("PCA model not found.")
    print("Training PCA using images in data/images...")
    create_embeddings()
    print("PCA training complete.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():

    if "image" not in request.files:
        return "No image uploaded"

    image = request.files["image"]

    if image.filename == "":
        return "No image selected"

    # Save uploaded image
    filename = secure_filename(image.filename)

    unique_name = str(uuid.uuid4()) + "_" + filename

    input_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    image.save(input_path)

    # Read controls from the website
    component = int(request.form.get("component", 0))
    strength = float(request.form.get("strength", 0))

    # Encode new image
    embedding = encode_new_image(input_path)

    # Modify embedding
    modified_embedding = embedding.copy()

    if 0 <= component < len(modified_embedding):
        modified_embedding[component] += strength

    # Decode modified embedding
    result = decode_embedding(modified_embedding)

    output_name = "result_" + unique_name + ".png"

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )

    result.save(output_path)

    return render_template(
        "index.html",
        original_image="/uploads/" + unique_name,
        result_image="/output/" + output_name,
        embedding=embedding.tolist(),
        component=component,
        strength=strength
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


if __name__ == "__main__":
    app.run(debug=True)