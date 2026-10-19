
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    HRFlowable,
)


def _format_value(value):
    """Convert feature values into readable PDF text."""
    try:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    except Exception:
        return str(value)


def _add_image(story, image_path, max_width=15 * cm, max_height=10 * cm):
    """Add an image to the PDF if the file exists."""
    if not image_path or not os.path.isfile(image_path):
        return

    try:
        img = Image(image_path)
        scale = min(
            max_width / img.imageWidth,
            max_height / img.imageHeight,
            1.0,
        )
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        story.append(img)
        story.append(Spacer(1, 0.3 * cm))
    except Exception:
        story.append(Paragraph(
            "Image could not be added to this report.",
            getSampleStyleSheet()["BodyText"],
        ))


def generate_pdf_report(
    output_path,
    image_path,
    prediction,
    features,
    roi_info=None,
    fft_path=None,
    heatmap_path=None,
):
    """Generate a PDF report for one PneumoAI analysis."""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="PneumoAI Chest X-Ray Analysis Report",
        author="PneumoAI",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#183153"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#183153"),
        spaceBefore=12,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#555555"),
    ))

    story = []

    story.append(Paragraph("PneumoAI", styles["ReportTitle"]))
    story.append(Paragraph(
        "Chest X-Ray Analysis Report",
        ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontSize=12,
            spaceAfter=10,
        ),
    ))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor("#D5DEE9"),
    ))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Report Details", styles["SectionHeading"]))
    details = [
        ["Generated on", datetime.now().strftime("%d %B %Y, %I:%M %p")],
        ["Model", "VGG16 (trained research model)"],
        ["Analysis pipeline", "Image preprocessing, DSP, segmentation and feature extraction"],
    ]
    details_table = Table(details, colWidths=[4.2 * cm, 12 * cm])
    details_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#222222")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DEE9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(details_table)

    story.append(Paragraph("Model Prediction", styles["SectionHeading"]))
    label = prediction.get("label", "Unavailable")
    confidence = prediction.get("confidence", "Unavailable")

    prediction_data = [
        ["Predicted class", str(label)],
        ["Model confidence score", f"{confidence}%" if confidence != "Unavailable" else confidence],
    ]
    prediction_table = Table(prediction_data, colWidths=[5 * cm, 11.2 * cm])
    prediction_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF3F8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DEE9")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(prediction_table)

    story.append(Paragraph("Uploaded Chest X-Ray", styles["SectionHeading"]))
    _add_image(story, image_path)

    if fft_path:
        story.append(Paragraph("DSP: 2D FFT Spectrum", styles["SectionHeading"]))
        _add_image(story, fft_path, max_width=13 * cm, max_height=8 * cm)

    if heatmap_path:
        story.append(Paragraph("Explainable AI: Grad-CAM", styles["SectionHeading"]))
        _add_image(story, heatmap_path, max_width=13 * cm, max_height=8 * cm)
        story.append(Paragraph(
            "Grad-CAM highlights regions that influenced the model output. "
            "It does not confirm the presence or location of disease.",
            styles["SmallNote"],
        ))

    story.append(Paragraph("Extracted Image Features", styles["SectionHeading"]))

    feature_rows = [["Feature", "Value"]]
    if isinstance(features, dict) and features:
        for key, value in features.items():
            feature_rows.append([
                Paragraph(str(key), styles["SmallNote"]),
                Paragraph(_format_value(value), styles["SmallNote"]),
            ])
    else:
        feature_rows.append(["Features", "No feature values available"])

    feature_table = Table(
        feature_rows,
        colWidths=[8 * cm, 8.2 * cm],
        repeatRows=1,
    )
    feature_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE9")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(feature_table)

    if roi_info:
        story.append(Paragraph("Segmentation / ROI Information", styles["SectionHeading"]))
        story.append(Paragraph(
            _format_value(roi_info),
            styles["SmallNote"],
        ))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Important Medical Disclaimer", styles["SectionHeading"]))
    story.append(Paragraph(
        "PneumoAI is an educational research prototype, not a clinically "
        "validated diagnostic system. Model predictions and confidence scores "
        "may be incorrect and must not be used for diagnosis or treatment "
        "decisions. Consult a qualified medical professional for medical assessment.",
        styles["SmallNote"],
    ))

    doc.build(story)
    return output_path
