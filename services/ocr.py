import os
import concurrent.futures
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


def correct_orientation(image):
    try:
        osd = pytesseract.image_to_osd(image, config="--psm 0 --oem 1")
        import re as _re
        angle = int(_re.search(r"Orientation in degrees: (\d+)", osd).group(1))
        if angle in (90, 180, 270):
            image = image.rotate(-angle, expand=True, fillcolor=(255, 255, 255))
    except Exception:
        pass
    return image


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
    image = correct_orientation(image)
    return _ocr_image_to_text(image)


def extract_words_from_image(image_path: str) -> list[dict]:
    if not tesseract_available:
        return []
    image = Image.open(image_path)
    image = correct_orientation(image)
    scale = max(1, 1200 // max(image.size))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    processed = preprocess_image_light(image)
    data = pytesseract.image_to_data(processed, config="--psm 3 --oem 1", output_type=pytesseract.Output.DICT)
    words = []
    img_w, img_h = processed.size
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
        if text and len(text) > 1 and conf > 20:
            words.append({
                "text": text,
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
                "conf": conf,
                "page_w": img_w,
                "page_h": img_h,
            })
    return words


def _ocr_image_to_text(image) -> str:
    scale = max(1, 1200 // max(image.size))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    processed = preprocess_image_light(image)
    text = pytesseract.image_to_string(processed, config="--psm 3 --oem 1").strip()
    if text:
        return text
    processed = preprocess_image_aggressive(image)
    text = pytesseract.image_to_string(processed, config="--psm 6 --oem 1").strip()
    return text


def _ocr_pil_image(pil_image):
    try:
        pil_image = correct_orientation(pil_image)
        return _ocr_image_to_text(pil_image)
    except Exception:
        return ""


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
        page_images = [page.to_image(resolution=150).original for page in pdf.pages]

    text_parts = []
    workers = min(4, len(page_images)) if len(page_images) > 1 else 1
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_ocr_pil_image, page_images)
            for text in results:
                if text:
                    text_parts.append(text)
    else:
        for img in page_images:
            text = _ocr_pil_image(img)
            if text:
                text_parts.append(text)

    return "\n".join(text_parts).strip()


def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".webp"):
        return extract_text_from_image(file_path)
    else:
        return ""
