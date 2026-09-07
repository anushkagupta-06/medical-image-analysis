from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import cv2
import numpy as np

from preprocessing.preprocess import preprocess_pipeline
from segmentation.segment import segment_pipeline
from models.feature_extractor import extract_all_features


app = Flask(__name__)

# ---------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp"
}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def allowed_file(filename):
    """Check whether uploaded file has a supported extension."""

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_image(image, filepath):
    """
    Save a NumPy/OpenCV image to disk.

    Handles grayscale, BGR and non-uint8 images.
    """

    if image is None:
        return False

    image_to_save = image.copy()

    # Convert floating-point images to uint8
    if image_to_save.dtype != np.uint8:

        image_to_save = cv2.normalize(
            image_to_save,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    return cv2.imwrite(filepath, image_to_save)


def save_pipeline_images(
    preprocessing_results,
    segmentation_results,
    session_folder
):
    """
    Save all preprocessing and segmentation outputs.

    Returns dictionaries containing paths that can
    be passed to the Flask templates.
    """

    preprocessing_images = {}

    segmentation_images = {}

    # -----------------------------------------------------
    # PREPROCESSING IMAGES
    # -----------------------------------------------------

    preprocessing_names = [
        "original",
        "grayscale",
        "gaussian_blur",
        "median_filter",
        "bilateral_filter",
        "denoised",
        "histogram_eq",
        "clahe",
        "sharpened"
    ]

    for name in preprocessing_names:

        if name in preprocessing_results:

            filename = f"{name}.png"

            filepath = os.path.join(
                session_folder,
                filename
            )

            if save_image(
                preprocessing_results[name],
                filepath
            ):
                preprocessing_images[name] = (
                    f"uploads/{os.path.basename(session_folder)}/{filename}"
                )

    # -----------------------------------------------------
    # SEGMENTATION IMAGES
    # -----------------------------------------------------

    segmentation_names = [
        "binary_threshold",
        "otsu_threshold",
        "adaptive_threshold",
        "canny_edges",
        "sobel_edges",
        "morphological",
        "watershed",
        "roi_mask",
        "roi_annotated"
    ]

    for name in segmentation_names:

        if name in segmentation_results:

            filename = f"{name}.png"

            filepath = os.path.join(
                session_folder,
                filename
            )

            if save_image(
                segmentation_results[name],
                filepath
            ):
                segmentation_images[name] = (
                    f"uploads/{os.path.basename(session_folder)}/{filename}"
                )

    return preprocessing_images, segmentation_images


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    # -----------------------------------------------------
    # CHECK FILE
    # -----------------------------------------------------

    if "file" not in request.files:

        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":

        return redirect(url_for("home"))

    if not allowed_file(file.filename):

        return redirect(url_for("home"))

    # -----------------------------------------------------
    # CREATE SESSION
    # -----------------------------------------------------

    session_id = uuid.uuid4().hex

    session_folder = os.path.join(
        app.config["UPLOAD_FOLDER"],
        session_id
    )

    os.makedirs(session_folder, exist_ok=True)

    # -----------------------------------------------------
    # SAVE ORIGINAL UPLOAD
    # -----------------------------------------------------

    extension = file.filename.rsplit(".", 1)[1].lower()

    uploaded_filename = f"uploaded.{extension}"

    uploaded_path = os.path.join(
        session_folder,
        uploaded_filename
    )

    file.save(uploaded_path)

    print("\n" + "=" * 60)
    print("PNEUMOAI ANALYSIS")
    print("=" * 60)

    # -----------------------------------------------------
    # 1. PREPROCESSING
    # -----------------------------------------------------

    print("\n[1/3] Running preprocessing...")

    preprocessing_results = preprocess_pipeline(
        uploaded_path
    )

    print("✓ Preprocessing completed")

    # -----------------------------------------------------
    # 2. SEGMENTATION
    # -----------------------------------------------------

    print("\n[2/3] Running segmentation...")

    grayscale = preprocessing_results["grayscale"]
    original = preprocessing_results["original"]

    segmentation_results = segment_pipeline(
        grayscale,
        original
    )

    print("✓ Segmentation completed")

    # -----------------------------------------------------
    # 3. FEATURE EXTRACTION
    # -----------------------------------------------------

    print("\n[3/3] Extracting features...")

    features = extract_all_features(grayscale)

    print(
        f"✓ Extracted {len(features)} numerical features"
    )

    # -----------------------------------------------------
    # SAVE ALL PROCESSED IMAGES
    # -----------------------------------------------------

    print("\nSaving processed images...")

    preprocessing_images, segmentation_images = (
        save_pipeline_images(
            preprocessing_results,
            segmentation_results,
            session_folder
        )
    )

    print("✓ Processed images saved")

    # -----------------------------------------------------
    # ROI
    # -----------------------------------------------------

    roi_info = segmentation_results.get("roi_info")

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)

    return render_template(
        "results.html",

        session_id=session_id,

        preprocessing_images=preprocessing_images,

        segmentation_images=segmentation_images,

        features=features,

        roi_info=roi_info,

        prediction=None
    )


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)