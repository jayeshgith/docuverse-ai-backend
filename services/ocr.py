import os
import re
import concurrent.futures
from pathlib import Path
import pytesseract
from pdfplumber import open as open_pdf
from PIL import Image, ImageEnhance

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

MAX_PDF_PAGES = 20
OCR_RESIZE_MIN = 800
PDF_RENDER_DPI = 100

# Single-file cache avoids redoing orientation/resize/preprocess
# when both text and word extraction run on the same image.
_ocr_cache = {
    "path": None,
    "oriented": None,
    "scaled": None,
    "preprocessed": None,
}


def _cache_get(image_path):
    if _ocr_cache["path"] == image_path:
        return _ocr_cache["oriented"], _ocr_cache["scaled"], _ocr_cache["preprocessed"]
    return None


def _cache_set(image_path, oriented, scaled, preprocessed):
    _ocr_cache["path"] = image_path
    _ocr_cache["oriented"] = oriented
    _ocr_cache["scaled"] = scaled
    _ocr_cache["preprocessed"] = preprocessed


def correct_orientation(image):
    try:
        osd = pytesseract.image_to_osd(image, config="--psm 0 --oem 1")
        angle = int(re.search(r"Orientation in degrees: (\d+)", osd).group(1))
        if angle in (90, 180, 270):
            image = image.rotate(-angle, expand=True, fillcolor=(255, 255, 255))
    except Exception:
        pass
    return image


def preprocess_image(image):
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.5)


def _resize_if_small(image):
    scale = max(1, OCR_RESIZE_MIN // max(image.size))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    return image


def extract_text_from_image(image_path: str) -> str:
    if not tesseract_available:
        return ""
    cached = _cache_get(image_path)
    if cached:
        _, _, processed = cached
    else:
        image = Image.open(image_path)
        oriented = correct_orientation(image)
        scaled = _resize_if_small(oriented)
        processed = preprocess_image(scaled)
        _cache_set(image_path, oriented, scaled, processed)
    return pytesseract.image_to_string(processed, config="--psm 3 --oem 1").strip()


def extract_words_from_image(image_path: str) -> list[dict]:
    if not tesseract_available:
        return []
    cached = _cache_get(image_path)
    if cached:
        oriented, scaled, processed = cached
    else:
        image = Image.open(image_path)
        oriented = correct_orientation(image)
        scaled = _resize_if_small(oriented)
        processed = preprocess_image(scaled)
        _cache_set(image_path, oriented, scaled, processed)
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


def _ocr_pil_image(pil_image):
    try:
        img = _resize_if_small(pil_image)
        processed = preprocess_image(img)
        return pytesseract.image_to_string(processed, config="--psm 3 --oem 1").strip()
    except Exception:
        return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    with open_pdf(pdf_path) as pdf:
        text_parts = []
        for page in pdf.pages[:MAX_PDF_PAGES]:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text_parts.append(page_text.strip())

    combined = "\n".join(text_parts).strip()
    if combined:
        return combined

    if not tesseract_available:
        return ""

    with open_pdf(pdf_path) as pdf:
        pages = pdf.pages[:MAX_PDF_PAGES]
        page_images = [page.to_image(resolution=PDF_RENDER_DPI).original for page in pages]

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

    if len(pdf.pages) > MAX_PDF_PAGES:
        print(f"[OCR] Capped at {MAX_PDF_PAGES} pages (document has {len(pdf.pages)} total)")

    return "\n".join(text_parts).strip()


def extract_text(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".webp"):
        return extract_text_from_image(file_path)
    else:
        return ""
