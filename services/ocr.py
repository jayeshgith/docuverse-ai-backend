import os
import concurrent.futures
import threading
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


_cache_lock = threading.Lock()
_ocr_cache = {
    "file_path": None,
    "oriented_image": None,
    "processed_light_image": None,
    "raw_text": None,
    "words": None
}


def extract_text_from_image(image_path: str) -> str:
    if not tesseract_available:
        return ""
    
    global _ocr_cache
    with _cache_lock:
        if _ocr_cache["file_path"] == image_path and _ocr_cache["raw_text"] is not None:
            return _ocr_cache["raw_text"]
        new_file = _ocr_cache["file_path"] != image_path
        if new_file:
            _ocr_cache = {"file_path": image_path, "oriented_image": None, "processed_light_image": None, "raw_text": None, "words": None}

        if _ocr_cache["oriented_image"] is None:
            try:
                image = Image.open(image_path)
                _ocr_cache["oriented_image"] = image
            except Exception:
                return ""
        image = _ocr_cache["oriented_image"]

        if _ocr_cache["processed_light_image"] is None or new_file:
            scale = max(1, 800 // max(image.size))
            img = image.resize((image.width * scale, image.height * scale), Image.LANCZOS) if scale > 1 else image
            processed = preprocess_image_light(img)
            _ocr_cache["processed_light_image"] = processed
        processed = _ocr_cache["processed_light_image"]

    text = pytesseract.image_to_string(processed, config="--psm 6 --oem 1").strip()
    if not text:
        oriented = _ocr_cache.get("oriented_image") if _ocr_cache.get("file_path") == image_path else None
        if oriented:
            with _cache_lock:
                oriented_corrected = correct_orientation(oriented)
                scale = max(1, 800 // max(oriented_corrected.size))
                img = oriented_corrected.resize((oriented_corrected.width * scale, oriented_corrected.height * scale), Image.LANCZOS) if scale > 1 else oriented_corrected
                processed_agg = preprocess_image_aggressive(img)
            text = pytesseract.image_to_string(processed_agg, config="--psm 6 --oem 1").strip()

    with _cache_lock:
        if _ocr_cache["file_path"] == image_path:
            _ocr_cache["raw_text"] = text

    return text


def extract_words_from_image(image_path: str) -> list[dict]:
    if not tesseract_available:
        return []

    global _ocr_cache
    with _cache_lock:
        if _ocr_cache["file_path"] == image_path:
            if _ocr_cache["words"] is not None:
                return _ocr_cache["words"]
        else:
            _ocr_cache["file_path"] = image_path
            _ocr_cache["oriented_image"] = None
            _ocr_cache["processed_light_image"] = None
            _ocr_cache["raw_text"] = None
            _ocr_cache["words"] = None

        if _ocr_cache["oriented_image"] is None:
            try:
                image = Image.open(image_path)
                image = correct_orientation(image)
                _ocr_cache["oriented_image"] = image
            except Exception:
                return []
        else:
            image = _ocr_cache["oriented_image"]

        if _ocr_cache["processed_light_image"] is None:
            scale = max(1, 1200 // max(image.size))
            if scale > 1:
                img_resized = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
            else:
                img_resized = image
            processed = preprocess_image_light(img_resized)
            _ocr_cache["processed_light_image"] = processed
        else:
            processed = _ocr_cache["processed_light_image"]

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

    with _cache_lock:
        if _ocr_cache["file_path"] == image_path:
            _ocr_cache["words"] = words

    return words


def _ocr_image_to_text(image) -> str:
    scale = max(1, 800 // max(image.size))
    img = image.resize((image.width * scale, image.height * scale), Image.LANCZOS) if scale > 1 else image
    processed = preprocess_image_light(img)
    text = pytesseract.image_to_string(processed, config="--psm 6 --oem 1").strip()
    if text:
        return text
    processed = preprocess_image_aggressive(img)
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
        page_images = [page.to_image(resolution=100).original for page in pdf.pages]

    text_parts = []
    workers = min(os.cpu_count() or 2, len(page_images)) if len(page_images) > 1 else 1
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_ocr_pil_image, img) for img in page_images]
            for f in concurrent.futures.as_completed(futures):
                text = f.result()
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
