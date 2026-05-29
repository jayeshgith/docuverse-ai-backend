import os
from pathlib import Path
import pytesseract
from pdfplumber import open as open_pdf
from PIL import Image, ImageEnhance, ImageFilter

tesseract_available = False
tesseract_cmd = os.environ.get("TESSERACT_CMD", "")
if not tesseract_cmd or not os.path.exists(tesseract_cmd):
    for candidate in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]:
        if os.path.exists(candidate):
            tesseract_cmd = candidate
            break
if os.path.exists(tesseract_cmd):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    tesseract_available = True


def preprocess_image_light(image):
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)


def preprocess_image_aggressive(image):
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.point(lambda x: 0 if x < 160 else 255, "1")
    return image


def extract_text_from_image(image_path: str) -> str:
    if not tesseract_available:
        return ""
    image = Image.open(image_path)

    scale = max(1, 1200 // max(image.size))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)

    processed = preprocess_image_light(image)
    text = pytesseract.image_to_string(processed, config="--psm 3 --oem 3").strip()
    if text:
        return text

    processed = preprocess_image_aggressive(image)
    text = pytesseract.image_to_string(processed, config="--psm 6 --oem 3").strip()
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    with open_pdf(pdf_path) as pdf:
        text_parts = []
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())

    combined = "\n".join(text_parts).strip()
    if combined:
        return combined

    if not tesseract_available:
        return ""

    with open_pdf(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                img = page.to_image(resolution=200)
                processed = preprocess_image_light(img.original)
                page_text = pytesseract.image_to_string(processed, config="--psm 3 --oem 3").strip()
                if not page_text:
                    processed = preprocess_image_aggressive(img.original)
                    page_text = pytesseract.image_to_string(processed, config="--psm 6 --oem 3").strip()
                if page_text:
                    text_parts.append(page_text)
            except Exception:
                pass

    return "\n".join(text_parts).strip()


def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".webp"):
        return extract_text_from_image(file_path)
    else:
        return ""
