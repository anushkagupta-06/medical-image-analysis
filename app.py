
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
import numpy as np

from models.pneumonia_classifier import predict_xray
from models.feature_extractor import extract_all_features

from preprocessing.preprocess import preprocess_pipeline
from segmentation.segment import segment_pipeline

from utils.dsp_analysis import compute_fft
from utils.report_generator import generate_pdf_report


# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

ALLOWED_EXTENSIONS = {
    "png", "jpg", "jpeg", "tif", "tiff", "bmp"
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def allowed_file(filename):
    """Check whether the uploaded file has a supported extension."""
    return (
        bool(filename)
        and "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_image(image, filepath):
    """Save an OpenCV/NumPy image as a PNG file."""
    if image is None:
        return False

    image_to_save = image.copy()

    if image_to_save.dtype != np.uint8:
        if image_to_save.size == 0:
            return False

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
    """Save preprocessing and segmentation outputs for the report/UI."""

    preprocessing_images = {}
    segmentation_images = {}

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

    session_id = os.path.basename(session_folder)

    for name in preprocessing_names:
        if name in preprocessing_results:
            filename = f"{name}.png"
            filepath = os.path.join(session_folder, filename)

            if save_image(preprocessing_results[name], filepath):
                preprocessing_images[name] = (
                    f"uploads/{session_id}/{filename}"
                )

    for name in segmentation_names:
        if name in segmentation_results:
            filename = f"{name}.png"
            filepath = os.path.join(session_folder, filename)

            if save_image(segmentation_results[name], filepath):
                segmentation_images[name] = (
                    f"uploads/{session_id}/{filename}"
                )

    return preprocessing_images, segmentation_images


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# ANALYZE CHEST X-RAY
# =========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    # -----------------------------------------------------
    # 1. VALIDATE UPLOAD
    # -----------------------------------------------------

    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]

    if not file.filename:
        return redirect(url_for("home"))

    if not allowed_file(file.filename):
        return (
            "Unsupported file type. Upload PNG, JPG, JPEG, TIF, TIFF or BMP.",
            400
        )

    safe_filename = secure_filename(file.filename)
    extension = safe_filename.rsplit(".", 1)[1].lower()

    # -----------------------------------------------------
    # 2. CREATE UNIQUE SESSION FOLDER
    # -----------------------------------------------------

    session_id = uuid.uuid4().hex

    session_folder = os.path.join(
        app.config["UPLOAD_FOLDER"],
        session_id
    )

    os.makedirs(session_folder, exist_ok=True)

    uploaded_path = os.path.join(
        session_folder,
        f"uploaded.{extension}"
    )

    file.save(uploaded_path)

    print("\n" + "=" * 60)
    print("PNEUMOAI ANALYSIS")
    print("=" * 60)

    try:

        # -------------------------------------------------
        # 3. IMAGE PREPROCESSING
        # -------------------------------------------------

        print("\n[1/5] Running preprocessing...")

        preprocessing_results = preprocess_pipeline(uploaded_path)

        if (
            not preprocessing_results
            or preprocessing_results.get("grayscale") is None
        ):
            raise ValueError("Image preprocessing failed.")

        grayscale = preprocessing_results["grayscale"]
        original = preprocessing_results["original"]

        print("Preprocessing completed.")

        # -------------------------------------------------
        # 4. DSP / 2D FFT ANALYSIS
        # -------------------------------------------------

        print("\n[2/5] Running 2D FFT analysis...")

        fft_image = compute_fft(grayscale)

        fft_path = os.path.join(
            session_folder,
            "fft_spectrum.png"
        )

        if not save_image(fft_image, fft_path):
            fft_path = None
            print("Warning: FFT image could not be saved.")

        print("DSP analysis completed.")

        # -------------------------------------------------
        # 5. SEGMENTATION
        # -------------------------------------------------

        print("\n[3/5] Running segmentation...")

        segmentation_results = segment_pipeline(
            grayscale,
            original
        )

        print("Segmentation completed.")

        # -------------------------------------------------
        # 6. FEATURE EXTRACTION
        # -------------------------------------------------

        print("\nExtracting image features...")

        features = extract_all_features(grayscale)

        print(f"Extracted {len(features)} numerical features.")

        # -------------------------------------------------
        # 7. VGG16 PREDICTION + GRAD-CAM
        # -------------------------------------------------

        print("\n[4/5] Running VGG16 prediction...")

        heatmap_path = os.path.join(
            session_folder,
            "gradcam.png"
        )

        prediction = predict_xray(
            uploaded_path,
            heatmap_path
        )

        prediction["heatmap_url"] = (
            f"uploads/{session_id}/gradcam.png"
        )

        print("Predicted class:", prediction["label"])
        print("Confidence:", prediction["confidence"], "%")

        # -------------------------------------------------
        # 8. SAVE PIPELINE IMAGES
        # -------------------------------------------------

        preprocessing_images, segmentation_images = (
            save_pipeline_images(
                preprocessing_results,
                segmentation_results,
                session_folder
            )
        )

        print("Processed images saved.")

        roi_info = segmentation_results.get("roi_info")

        # -------------------------------------------------
        # 9. GENERATE PDF REPORT
        # -------------------------------------------------

        print("\n[5/5] Generating PDF report...")

        report_path = os.path.join(
            session_folder,
            "analysis_report.pdf"
        )

        generate_pdf_report(
            output_path=report_path,
            image_path=uploaded_path,
            prediction=prediction,
            features=features,
            roi_info=roi_info,
            fft_path=fft_path,
            heatmap_path=heatmap_path
        )

        report_url = (
            f"uploads/{session_id}/analysis_report.pdf"
        )

        print("PDF report generated.")
        print("\nAnalysis completed successfully.")
        print("=" * 60)

        # -------------------------------------------------
        # 10. DISPLAY RESULTS
        # -------------------------------------------------

        return render_template(
            "results.html",
            session_id=session_id,
            preprocessing_images=preprocessing_images,
            segmentation_images=segmentation_images,
            features=features,
            roi_info=roi_info,
            prediction=prediction,
            report_url=report_url
        )

    except Exception as error:
        app.logger.exception("PneumoAI analysis failed")

        return (
            "Analysis could not be completed. "
            "Check the Flask terminal for the error details.",
            500
        )


# =========================================================
# FILE SIZE ERROR
# =========================================================

@app.errorhandler(413)
def file_too_large(error):
    return "File too large. Please upload an image smaller than 16 MB.", 413


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
