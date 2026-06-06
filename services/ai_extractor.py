import os
import json
import re
from openai import OpenAI
from services.database import get_db

client = None
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    print("[WARN] OPENAI_API_KEY not set. AI extraction will use regex-only fallback which may miss fields.")

HARDCODED_CONFIGS: dict[str, dict] = {
    "passport": {
        "fields": ["document_type", "passport_number", "name", "dob", "nationality", "gender", "issue_date", "expiry_date", "place_of_birth", "place_of_issue", "address"],
        "required_fields": ["passport_number", "name", "dob"],
        "llm_hint": "Extract passport details. Passport number format is one letter followed by 7 digits (e.g. W23454542).",
        "confidence_threshold": 0.78,
    },
    "pan_card": {
        "fields": ["document_type", "pan_number", "name", "father_name", "dob"],
        "required_fields": ["pan_number", "name"],
        "llm_hint": "Extract Indian PAN card details. PAN number format is 5 uppercase letters + 4 digits + 1 uppercase letter (e.g. ABCDE1234F). The cardholder name is in all caps. Father's name is also present. DOB is in DD/MM/YYYY format. Common labels: 'Permanent Account Number', 'Name', 'Father's Name', 'Date of Birth'.",
        "confidence_threshold": 0.78,
    },
    "aadhaar_card": {
        "fields": ["document_type", "aadhaar_number", "name", "dob", "gender", "address", "mobile_number"],
        "required_fields": ["aadhaar_number", "name"],
        "llm_hint": "Extract Aadhaar card details. Aadhaar is 12 digits in groups of 4 (e.g. 1234 5678 9012).",
        "confidence_threshold": 0.78,
    },
    "invoice": {
        "fields": ["document_type", "invoice_number", "name", "vendor", "date", "total_amount"],
        "required_fields": ["invoice_number", "vendor", "total_amount"],
        "llm_hint": "Extract invoice details including invoice number, customer name, vendor, date, and total amount.",
        "confidence_threshold": 0.78,
    },
    "bill": {
        "fields": ["document_type", "bill_number", "vendor", "date", "total_amount", "name"],
        "required_fields": ["bill_number", "total_amount"],
        "llm_hint": "Extract bill/receipt details including bill number, vendor, date, and total amount.",
        "confidence_threshold": 0.78,
    },
    "resume": {
        "fields": ["document_type", "name", "email", "phone", "skills", "education", "experience_summary"],
        "required_fields": ["name"],
        "llm_hint": "Extract resume details including full name, email, phone, skills, education, and experience.",
        "confidence_threshold": 0.78,
    },
    "other": {
        "fields": ["document_type", "name", "document_number", "date", "email", "phone", "father_name", "holder_name", "card_number", "address", "dob"],
        "required_fields": [],
        "llm_hint": "Extract any document details found: name, document number, date, email, phone, father's name, holder name, card number, address, and date of birth. Look for labels like 'Name', 'Card Number', 'Account Number', 'Father's Name', 'Address', 'DOB', 'Phone', 'Email'.",
        "confidence_threshold": 0.0,
    },
}


def get_doc_config(doc_type: str, tenant_id: str = "default") -> dict:
    try:
        db = get_db()
        cfg = db.document_configs.find_one({
            "document_type": doc_type.lower(),
            "tenant_id": tenant_id.lower()
        })
        if not cfg and tenant_id.lower() != "default":
            cfg = db.document_configs.find_one({
                "document_type": doc_type.lower(),
                "tenant_id": "default"
            })
        if not cfg:
            cfg = db.document_configs.find_one({"doc_type": doc_type.lower(), "enabled": True})
            
        if cfg:
            fields_raw = cfg.get("fields", [])
            if fields_raw and isinstance(fields_raw[0], dict):
                fields_keys = [f["key"] for f in fields_raw]
                required = [f["key"] for f in fields_raw if f.get("is_required", True)]
                desc_str = ", ".join(f"'{f['key']}' ({f['description']})" for f in fields_raw)
                llm_hint = f"Extract details for {cfg.get('display_name', doc_type)}. Look for: {desc_str}."
                return {
                    "fields": fields_keys,
                    "required_fields": required,
                    "llm_hint": llm_hint,
                    "confidence_threshold": cfg.get("confidence_threshold", 0.78),
                    "raw_fields": fields_raw
                }
            else:
                return {
                    "fields": fields_raw,
                    "required_fields": cfg.get("required_fields", []),
                    "llm_hint": cfg.get("llm_hint", ""),
                    "confidence_threshold": cfg.get("confidence_threshold", 0.78),
                    "raw_fields": []
                }
    except Exception:
        pass

    fallback = HARDCODED_CONFIGS.get(doc_type, HARDCODED_CONFIGS["other"])
    return {
        "fields": fallback["fields"],
        "required_fields": fallback["required_fields"],
        "llm_hint": fallback["llm_hint"],
        "confidence_threshold": fallback["confidence_threshold"],
        "raw_fields": []
    }

PASSPORT_NUM_RE = r"(?:passport|pasport|paspOrt|passpOrt)\s*(?:no|number|#|\.)?\s*[:\-]\s*(?:[A-Z]\s*[0-9]\s*[0-9]\s*[0-9]\s*[0-9]\s*[0-9]\s*[0-9]\s*[0-9])"
PAN_RE = r"(?:pan\s*(?:no|number|#|\.|:)?\s*[:\-]\s*)?([A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*\d\s*\d\s*\d\s*\d\s*[A-Z])"
AADHAAR_RE = r"(\d{4}\s?\d{4}\s?\d{4})"
AADHAAR_RE_MULTI = r"(\d{4})\s*(\d{4})\s*(\d{4})"
NAME_RE = r"(?:name|n a m e|neme|ful|full name|given name|surname|applicant name|candidate name|student name|holder name)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$|\||email|\d{2}|[0-9])"
DOB_RE = r"(?:dob|d\.o\.b|d\.0\.b|date\s*of\s*birth|birth\s*date|date\s*of\s*birth|birth)\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
DATE_RE = r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})"
EMAIL_RE = r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})"
PHONE_RE = r"((?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"
VENDOR_RE = r"(?:vendor|seller|supplier|store|shop|company|merchant|vender|billed to|bill from|biled to)\s*[:\-]\s*([A-Za-z0-9\s\.'\-&]+?)(?:\n|$)"
TOTAL_RE = r"(?:total|grand total|total amount|amount due|net amount|balance due|sum|tota1|totai)\s*[:\-]?\s*[₹$]?\s*([\d,]+\.\d{2})"
NATIONALITY_RE = r"(?:nationality|citizenship|nati0nality)\s*[:\-]\s*([A-Za-z\s]+?)(?:\n|$|\d)"
GENDER_RE = r"(?:gender|sex|gender|sex)\s*[:\-]\s*(M|F|Male|Female|MALE|FEMALE|male|female)"
FATHER_RE = r"(?:father|father's name|father name|fathers name|f ather|fath er)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)"
ISSUE_RE = r"(?:issue date|date of issue|issued on|date of issuance|iss ue date)\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
EXPIRY_RE = r"(?:expiry date|date of expiry|valid until|expiration date|valid till|expir y date)\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
ADDRESS_RE = r"(?:address|residence|permanent address|addr ess|add ress|perman ent)\s*[:\-]\s*([\w\s,\.\-/#]+?)(?:\n{2,}|$)"
INVOICE_RE = r"(?:invoice\s*(?:no|number|#|\.)?\s*[:\-]\s*)([A-Z0-9\-/]+)"
BILL_RE = r"(?:bill\s*(?:no|number|#|\.)?\s*[:\-]\s*)([A-Z0-9\-/]+)"
DOC_NUM_RE = r"(?:document\s*(?:no|number|#|\.)?\s*[:\-]\s*)([A-Z0-9\-]+)"

HOLDER_NAME_RE = r"(?:holder\s*(?:name)?|card\s*holder|cardholder|account\s*holder|acc0unt)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)"
CARD_NUM_RE = r"(?:card\s*(?:no|number|#|\.)?|account\s*(?:no|number|#|\.)?|member\s*(?:no|number)?)\s*[:\-]\s*([A-Z0-9\-/\s]{8,20})"
ADDRESS_FALLBACK_RE = r"((?:door|street|road|colony|sector|phase|block|house|building|apt|flat|village|city|town|district|state|pincode|pin\s*code)[,\s]*[\w\s,\.\-/#]+(?:\n|$))"

PLACE_OF_BIRTH_RE = r"(?:place\s*of\s*birth|pob|birth\s*place|p l a c e)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)"
PLACE_OF_ISSUE_RE = r"(?:place\s*of\s*issue|poi|issue\s*place)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)"
MOBILE_RE = r"(?:mobile|phone|contact|mobile\s*number|phone\s*number|telephone|m0bile|ph one)\s*[:\-]\s*(\+?\d[\d\s\-()]{7,15})"

PAN_CLEAN_RE = re.compile(r"[^A-Z0-9]")
LABEL_EXCLUDE = r"(pan|permanent|perm anent|account|acc0unt|number|num ber|income|tax|govt|india|date|birth|father|mother|signature|name|gender|address|dob|issue|expiry|nationality|phone|email|holder|card|aadhaar|uidai|resident)"
SEP_NL = r"\s*[:\-]?\s*\n\s*"


def detect_document_type(raw_text, tenant_id="default"):
    t = raw_text.lower()

    # Check configured document types from MongoDB first
    try:
        db = get_db()
        configured = list(db.document_configs.find({
            "$or": [{"tenant_id": tenant_id.lower()}, {"tenant_id": "default"}]
        }))
        seen = set()
        for cfg in configured:
            slug = cfg.get("document_type", "").lower()
            if slug in seen:
                continue
            seen.add(slug)
            display = cfg.get("display_name", "").lower()
            # Match by slug or display name in the raw text
            if slug.replace("_", " ") in t or (display and display in t):
                print(f"[INFO] detect_document_type: matched configured type '{slug}'")
                return slug
            # Match by regex patterns in configured fields
            for f in cfg.get("fields", []):
                pattern = f.get("regex_pattern")
                if pattern:
                    try:
                        if re.search(pattern, raw_text, re.IGNORECASE):
                            print(f"[INFO] detect_document_type: matched '{slug}' via field regex '{f.get('key')}'")
                            return slug
                    except Exception:
                        pass
    except Exception as e:
        print(f"[WARN] detect_document_type: config lookup error: {e}")

    # Fall back to hardcoded classifiers
    score = sum(1 for w in ["resume", "curriculum vitae", "cv", "experience", "skills", "education", "objective"] if w in t)
    if score >= 3:
        return "resume"
    if any(w in t for w in ["passport", "pasport", "pasp0rt", "passp0rt", "passport no", "passport number"]):
        return "passport"
    if any(w in t for w in ["pan card", "permanent account number", "perm anent", "income tax", "income tax department", "govt of india", "pancard"]):
        return "pan_card"
    if any(w in t for w in ["aadhaar", "aadhar", "adh aar", "uidai"]):
        return "aadhaar_card"
    if any(w in t for w in ["invoice", "tax invoice", "invoi ce", "inv no"]):
        return "invoice"
    if any(w in t for w in ["bill", "receipt", "recipt", "payment", "total due", "amount due"]):
        return "bill"

    cleaned = PAN_CLEAN_RE.sub("", raw_text)
    pan_match = re.search(r"[A-Z]{5}\d{4}[A-Z]", cleaned)
    if pan_match:
        return "pan_card"

    if re.search(PASSPORT_NUM_RE, raw_text):
        return "passport"
    if re.search(AADHAAR_RE, raw_text) or re.search(AADHAAR_RE_MULTI, raw_text):
        return "aadhaar_card"
    return "other"


def extract_fields_rule_based(raw_text, doc_type):
    fields = {"document_type": doc_type.replace("_", " ").title()}

    def get(regex):
        m = re.search(regex, raw_text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
        if r"\s*[:\-]\s*" in regex:
            alt = regex.replace(r"\s*[:\-]\s*", SEP_NL)
            m = re.search(alt, raw_text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).strip()
        return None

    if doc_type == "passport":
        fields["passport_number"] = get(PASSPORT_NUM_RE) or get(r"([A-Z][0-9]{7})")
        fields["name"] = get(NAME_RE)
        fields["dob"] = get(DOB_RE) or get(DATE_RE)
        fields["nationality"] = get(NATIONALITY_RE)
        fields["gender"] = get(GENDER_RE)
        fields["issue_date"] = get(ISSUE_RE)
        fields["expiry_date"] = get(EXPIRY_RE)
        fields["place_of_birth"] = get(PLACE_OF_BIRTH_RE)
        fields["place_of_issue"] = get(PLACE_OF_ISSUE_RE)
        fields["address"] = get(ADDRESS_RE)
        if not fields.get("name"):
            m = re.search(r"(?:mr\.|mrs\.|ms\.|shri|smt)\s+([A-Z][A-Za-z\s]+?)(?:\n|$)", raw_text, re.IGNORECASE)
            if m:
                fields["name"] = m.group(1).strip()

    elif doc_type == "pan_card":
        pan_raw = get(PAN_RE)
        if pan_raw:
            fields["pan_number"] = PAN_CLEAN_RE.sub("", pan_raw)

        fields["name"] = get(NAME_RE)
        if not fields.get("name"):
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                if re.search(r"(pan|permanent account|income tax|govt)", line, re.I):
                    for j in range(i + 1, min(i + 6, len(lines))):
                        candidate = lines[j].strip()
                        if (not re.search(LABEL_EXCLUDE, candidate, re.I)
                                and re.match(r"^[A-Z][A-Za-z\s.'-]{4,40}$", candidate)
                                and " " in candidate):
                            fields["name"] = candidate
                            break
                    break

        fields["father_name"] = get(FATHER_RE)
        if not fields.get("father_name"):
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                if re.search(r"(father|fathers)", line, re.I):
                    for j in range(i + 1, min(i + 3, len(lines))):
                        candidate = lines[j].strip()
                        if (not re.search(LABEL_EXCLUDE, candidate, re.I)
                                and re.match(r"^[A-Z][A-Za-z\s.'-]{4,40}$", candidate)
                                and " " in candidate):
                            fields["father_name"] = candidate
                            break
                    break

        fields["dob"] = get(DOB_RE) or get(DATE_RE)
        if not fields.get("dob"):
            m = re.search(r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", raw_text)
            if m:
                fields["dob"] = m.group(1).strip()

    elif doc_type == "aadhaar_card":
        aadhaar_raw = get(AADHAAR_RE)
        if not aadhaar_raw:
            m = re.search(AADHAAR_RE_MULTI, raw_text, re.IGNORECASE | re.DOTALL)
            if m:
                aadhaar_raw = f"{m.group(1)} {m.group(2)} {m.group(3)}"
        if aadhaar_raw:
            fields["aadhaar_number"] = re.sub(r"\s+", " ", aadhaar_raw)

        fields["name"] = get(NAME_RE)
        if not fields.get("name"):
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                if re.search(r"(aadhaar|uidai)", line, re.I):
                    for j in range(i + 1, min(i + 5, len(lines))):
                        candidate = lines[j].strip()
                        if (not re.search(LABEL_EXCLUDE, candidate, re.I)
                                and re.match(r"^[A-Za-z][A-Za-z\s.'-]{2,40}$", candidate)
                                and not re.match(r"^\d", candidate)):
                            fields["name"] = candidate
                            break
                    break

        if not fields.get("name"):
            m = re.search(r"(?:aadhaar|uidai|aadhar)\s*\n\s*([A-Za-z\s.'-]+?)\s*\n\s*(?:dob|date|birth|\d{2}/\d{2})", raw_text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if not re.search(LABEL_EXCLUDE, candidate, re.I) and len(candidate) >= 3:
                    fields["name"] = candidate

        if not fields.get("name"):
            m = re.search(r"(?:name|full name|applicant|holder)\s*[:.\-]?\s*([A-Za-z\s.'-]+?)(?:\n|$)", raw_text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip()
                if len(candidate) >= 3:
                    fields["name"] = candidate

        fields["dob"] = get(DOB_RE) or get(DATE_RE)
        if not fields.get("dob"):
            m = re.search(r"\b(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})\b", raw_text)
            if m:
                fields["dob"] = m.group(1).strip()

        fields["gender"] = get(GENDER_RE)
        if not fields.get("gender"):
            m = re.search(r"\b(Male|Female|MALE|FEMALE|M|F)\b", raw_text)
            if m:
                fields["gender"] = m.group(0).title()

        fields["address"] = get(ADDRESS_RE)

        if not fields.get("address"):
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            addr_start = -1
            for i, line in enumerate(lines):
                if re.search(r"(add|addr|address|residence|permanent)", line, re.I):
                    addr_start = i
                    break
            if addr_start >= 0:
                addr_lines = []
                for j in range(addr_start + 1, min(addr_start + 8, len(lines))):
                    line = lines[j]
                    if re.search(r"(mobile|phone|email|aadhaar|uidai)", line, re.I):
                        break
                    if len(line) > 5:
                        addr_lines.append(line)
                if addr_lines:
                    fields["address"] = ", ".join(addr_lines)[:200]

        fields["mobile_number"] = get(MOBILE_RE) or get(PHONE_RE)

    elif doc_type in ("invoice", "bill"):
        fields["invoice_number"] = get(INVOICE_RE)
        fields["bill_number"] = get(BILL_RE)
        fields["vendor"] = get(VENDOR_RE)
        fields["date"] = get(DATE_RE)
        fields["total_amount"] = get(TOTAL_RE)
        fields["name"] = get(NAME_RE)

    elif doc_type == "resume":
        fields["name"] = get(NAME_RE)
        fields["email"] = get(EMAIL_RE)
        fields["phone"] = get(PHONE_RE)
        m = re.search(r"(?:skills|technical skills|core competencies)[:\s]*\n?([\w\s,\.#+\n]+?)(?:\n{2,}|education|experience|$)", raw_text, re.IGNORECASE | re.DOTALL)
        if m:
            fields["skills"] = m.group(1).strip()[:200]
        m = re.search(r"(?:education|academic|qualification)[:\s]*\n?([\w\s,\.\-\(\)\n]+?)(?:\n{2,}|experience|skills|$)", raw_text, re.IGNORECASE | re.DOTALL)
        if m:
            fields["education"] = m.group(1).strip()[:300]
        m = re.search(r"(?:experience|work experience|professional experience)[:\s]*\n?([\w\s,\.\-\(\)\n]+?)(?:\n{2,}|education|skills|$)", raw_text, re.IGNORECASE | re.DOTALL)
        if m:
            fields["experience_summary"] = m.group(1).strip()[:300]

    else:
        fields["name"] = get(NAME_RE)
        if not fields.get("name"):
            fields["name"] = get(HOLDER_NAME_RE)
        fields["holder_name"] = get(HOLDER_NAME_RE)
        fields["document_number"] = get(DOC_NUM_RE) or get(CARD_NUM_RE) or get(PAN_RE) or get(PASSPORT_NUM_RE)
        fields["card_number"] = get(CARD_NUM_RE)
        fields["date"] = get(DATE_RE)
        fields["email"] = get(EMAIL_RE)
        fields["phone"] = get(PHONE_RE)
        fields["father_name"] = get(FATHER_RE)
        fields["address"] = get(ADDRESS_RE) or get(ADDRESS_FALLBACK_RE)
        fields["dob"] = get(DOB_RE) or get(DATE_RE)

    return {k: v for k, v in fields.items() if v}


def extract_with_openai(raw_text, doc_type, config=None):
    if config is None:
        config = get_doc_config(doc_type)
    schema = config["fields"]
    hint = config.get("llm_hint", "")
    prompt = f"""{hint}

Return ONLY a JSON object with these exact keys: {json.dumps(schema)}
Set missing fields to null. No extra keys.

OCR text from document:
{raw_text[:8000]}"""

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You extract structured data from OCR text. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.05, max_tokens=600,
            timeout=15,
        )
        c = r.choices[0].message.content.strip()
        c = re.sub(r"^```(?:json)?\s*|\s*```$", "", c)
        return {k: v for k, v in json.loads(c).items() if v is not None}
    except Exception:
        return None


def extract_fields_rule_based_dynamic(raw_text, config_raw_fields) -> dict:
    fields = {}
    for f in config_raw_fields:
        key = f.get("key")
        pattern = f.get("regex_pattern")
        if key and pattern:
            try:
                m = re.search(pattern, raw_text, re.IGNORECASE | re.MULTILINE)
                if m:
                    fields[key] = m.group(1).strip() if m.groups() else m.group(0).strip()
            except Exception:
                pass
    return fields


def extract_fields(raw_text, tenant_id="default"):
    if not raw_text or len(raw_text.strip()) < 5:
        print("[WARN] extract_fields: text too short, skipping")
        return {}, {}, 0.0

    doc_type = detect_document_type(raw_text, tenant_id)
    print(f"[INFO] detect_document_type -> {doc_type}")

    config = get_doc_config(doc_type, tenant_id)
    print(f"[INFO] Config loaded: {len(config.get('fields', []))} fields, {len(config.get('required_fields', []))} required {config.get('required_fields', [])}")

    rule_fields = extract_fields_rule_based(raw_text, doc_type)
    print(f"[INFO] Regex matched {len(rule_fields)} fields: {list(rule_fields.keys())}")

    if config.get("raw_fields"):
        dynamic_rule_fields = extract_fields_rule_based_dynamic(raw_text, config["raw_fields"])
        if dynamic_rule_fields:
            print(f"[INFO] Dynamic regex matched {len(dynamic_rule_fields)} extra fields: {list(dynamic_rule_fields.keys())}")
        rule_fields.update(dynamic_rule_fields)

    required = config.get("required_fields", [])
    threshold = config.get("confidence_threshold", 0.78)

    # Short-circuit check: if all required fields are captured by regex, skip OpenAI
    if required and all(rule_fields.get(f) for f in required) and len(rule_fields) >= len(required):
        scores = {k: 0.85 for k in rule_fields}
        overall = round(sum(scores.values()) / len(scores), 2) if scores else 0.0
        if overall >= threshold and overall > 0:
            print(f"[INFO] RULES-FIRST SHORT-CIRCUIT: Skipped OpenAI API for {doc_type}!")
            return rule_fields, scores, overall

    missing = [f for f in required if not rule_fields.get(f)]
    if missing:
        print(f"[INFO] Missing required fields (need OpenAI): {missing}")

    ai_fields = None
    if client:
        try:
            print(f"[INFO] Calling OpenAI gpt-4o-mini for {doc_type} (regex got {len(rule_fields)} fields)...")
            ai_fields = extract_with_openai(raw_text, doc_type, config)
            print(f"[INFO] OpenAI returned {len(ai_fields) if ai_fields else 0} fields")
        except Exception as e:
            print(f"[ERROR] OpenAI error: {type(e).__name__}: {e}")
            ai_fields = None
    else:
        print("[WARN] OpenAI client not configured (no API key). Using regex-only results.")

    merged, scores = {}, {}
    keys = set(list(rule_fields.keys()) + (list(ai_fields.keys()) if ai_fields else []))

    for k in keys:
        rv = rule_fields.get(k)
        av = ai_fields.get(k) if ai_fields else None
        if rv and av:
            if rv.strip().lower() == av.strip().lower():
                merged[k], scores[k] = rv, 0.95
            elif len(av) >= len(rv):
                merged[k], scores[k] = av, 0.85
            else:
                merged[k], scores[k] = rv, 0.78
        elif av:
            merged[k], scores[k] = av, round(min(0.88, 0.70 + len(av.strip()) * 0.01), 2)
        elif rv:
            merged[k], scores[k] = rv, 0.78

    overall = round(sum(scores.values()) / len(scores), 2) if scores else 0.0
    return merged, scores, overall
