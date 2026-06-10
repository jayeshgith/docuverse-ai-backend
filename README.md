# DocuVerse AI — Document Extraction & Chat API

Welcome to the brain of **DocuVerse AI**. 

I built this backend as a core part of my transition from a Next.js Full-Stack Developer into the world of AI Engineering. Rather than just copy-pasting an AI wrapper, I wanted to design a production-ready, highly thoughtful document intelligence system. 

This API solves a real-world problem: manual data entry from physical documents (like IDs, invoices, and resumes) is slow, frustrating, and prone to human error. DocuVerse automates this by reading, understanding, and extracting structured information from raw files, while also letting users interactively chat with their documents.

---

## 🧠 The Design Choices & Architecture Thinking

When designing this backend, I made several deliberate system architecture choices:

### 1. The Hybrid Extraction Strategy (Cost & Speed Optimization)
Large Language Models (LLMs) are incredibly smart, but they are also slow, expensive, and can occasionally hallucinate. 
To build an efficient system, I designed a **Blended/Hybrid Extraction Pipeline**:
* **Deterministic Rules (Regex)**: For highly structured data with standard formats (like Indian PAN cards, Aadhaar cards, passport numbers, dates, and emails), the system uses optimized Python regular expressions. This is **100% accurate, completely free, and takes milliseconds**.
* **LLM Fallback (GPT-4o-mini)**: For complex, unstructured text (like a resume, an invoice layout, or a business bill), the system feeds the parsed text to GPT-4o-mini using OpenAI's **Structured Outputs**. This gives us clean, reliable JSON schemas.
* **Blended Confidence Score**: The system automatically merges these two extraction lists and calculates a confidence score, ensuring the user gets the best possible data.

### 2. Solving the OCR Quality Problem
In the real world, users don't upload perfect scans. They take blurry, dark photos of their IDs with their phones. 
To handle this, I built an **Image Preprocessing Pipeline** using Pillow (PIL) before sending files to Tesseract OCR:
* It converts the image to grayscale to remove background noise.
* It dynamically scales the image up if it's too small.
* It increases contrast and applies sharpening filters to make text stand out.
This simple engineering step dramatically boosted our OCR read success rates on low-quality smartphone uploads.

### 3. Data Privacy and Local AI (RAG Chatbot)
Document data is highly personal—passports, financial bills, and resumes contain private information. Sending all of this to external APIs for simple search queries is a security risk.
To solve this, I built a local **Retrieval-Augmented Generation (RAG)** chatbot using **Ollama**:
* It uses a local **Phi-3 (mini)** model running right on the system.
* The extracted text of the document is passed as context to the local model.
* Users can ask questions about their documents securely, knowing that their data never leaves their local network.

---

## 🛠️ The Tech Stack
* **Framework**: **FastAPI** (High-speed, asynchronous Python framework with auto-generated Swagger UI).
* **Database**: **MongoDB** (storing document metadata and raw file binary payloads).
* **OCR**: **PyTesseract** (Google's Tesseract Engine) & **pdfplumber** (for digital PDF parsing).
* **AI Layer**: **OpenAI SDK** & local **Ollama (Phi-3)**.
* **Validation**: **Pydantic** (data validation schemas, similar to Zod in TypeScript).

---

## 🚀 Quick Setup (How to run it locally)

### Prerequisites
Make sure you have **Python 3.10+** and **Tesseract OCR** installed on your system.

### 1. Clone & Navigate
```bash
git clone https://github.com/jayeshgith/docuverse-ai-backend.git
cd docuverse-ai-backend
```

### 2. Set Up Virtual Environment
On Windows:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root backend directory:
```env
MONGODB_URL=your_mongodb_connection_string
DATABASE_NAME=your_database_name
OPENAI_API_KEY=your_openai_key
SMTP_EMAIL=your_email
SMTP_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### 5. Launch the Server
```bash
uvicorn main:app --reload --port 8000
```
* Visit `http://localhost:8000/docs` in your browser to view and test the API endpoints interactively!
