# =======================
# Mount Google Drive
# =======================
from google.colab import drive
drive.mount('/content/drive')

import requests
import pandas as pd
import time
import io

# =======================
# Try alternative PDF libraries first (avoiding fitz/PyMuPDF issues)
# =======================
PDF_LIBRARY = None

# Option 1: Try pdfplumber (most reliable)
try:
    import pdfplumber
    PDF_LIBRARY = "pdfplumber"
    print("Using pdfplumber for PDF text extraction")
except ImportError:
    pass

# Option 2: Try pypdf (lightweight alternative)
if not PDF_LIBRARY:
    try:
        from pypdf import PdfReader
        PDF_LIBRARY = "pypdf"
        print("Using pypdf for PDF text extraction")
    except ImportError:
        pass

# Option 3: Fallback to PyMuPDF only if others fail
if not PDF_LIBRARY:
    try:
        import fitz  # PyMuPDF
        PDF_LIBRARY = "pymupdf"
        print("Using PyMuPDF for PDF text extraction (fallback)")
    except ImportError:
        raise ImportError("No PDF library found. Install one with: pip install pdfplumber OR pip install pypdf OR pip install pymupdf")

HEADERS = {"User-Agent": "Mozilla/5.0"}

def download_pdf(pdf_url):
    try:
        response = requests.get(pdf_url, headers=HEADERS)
        if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
            return response.content
        print(f"Failed to download PDF: {pdf_url} (Status code: {response.status_code})")
        return None
    except Exception as e:
        print(f"Error downloading PDF {pdf_url}: {e}")
        return None

def extract_text_from_pdf(pdf_bytes):
    """Extract text using the available PDF library."""
    try:
        if PDF_LIBRARY == "pdfplumber":
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip() if text.strip() else "No text found"

        elif PDF_LIBRARY == "pypdf":
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip() if text.strip() else "No text found"

        elif PDF_LIBRARY == "pymupdf":
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text if text.strip() else "No text found"

    except Exception as e:
        print(f"Error extracting text with {PDF_LIBRARY}: {e}")
        return "Error extracting text"


# =======================
# List of DSpace bitstream URLs
# =======================
pdf_urls = [
    "https://dspace.unza.zm/server/api/core/bitstreams/a60d36ec-0543-4b83-88fe-bc5f4ae5d6f4/content",
    "https://dspace.unza.zm/server/api/core/bitstreams/ab25b16d-864e-4081-80e8-6e78dfe18539/content",
    "https://dspace.unza.zm/server/api/core/bitstreams/b9734d51-b347-46e3-8902-3dfec0217103/content",
    "https://dspace.unza.zm/server/api/core/bitstreams/5042acf2-8dae-497f-9404-8cad36ddc3d8/content",
    "https://dspace.unza.zm/server/api/core/bitstreams/af81ba95-83f7-48d6-9adf-a4013068da2c/content",
    "https://dspace.unza.zm/server/api/core/bitstreams/40365a72-ca30-486e-99c9-842868623c91/content",
    "https://dspace.unza.zm/server/api/core/bitstreams/6c64642f-1de9-42eb-914f-31f6f4a57d22/content",
    "https://dspace.unza.zm/server/api/core/bitstreams/2fb18425-ae79-44a9-be07-d3f897e7915e/content",
    "https://dspace.unza.zm/server/api/core/bitstreams/c9349b9c-5cf4-400f-bcdb-ae1631e19955/content"
]

print(f"Starting PDF text extraction using {PDF_LIBRARY}")
print("=" * 60)

results = []

for i, pdf_url in enumerate(pdf_urls, 1):
    print(f"Processing {i}/{len(pdf_urls)}: {pdf_url}")
    pdf_bytes = download_pdf(pdf_url)
    if pdf_bytes:
        print(f"  Downloaded PDF ({len(pdf_bytes):,} bytes)")
        text = extract_text_from_pdf(pdf_bytes)
        print(f"  Extracted {len(text):,} characters of text")
    else:
        text = "Failed to download PDF"
        print("  Failed to download PDF")

    results.append({
        "PDF_URL": pdf_url,
        "Extracted_Text": text
    })

    time.sleep(1)  # Pause between requests
    print()

print("=" * 60)
df = pd.DataFrame(results)

# =======================
# Save directly into your Drive folder
# =======================
save_path = "/content/drive/MyDrive/misc-unza25-csc4792-project_team30/references.csv"
df.to_csv(save_path, index=False)

# =======================
# Summary
# =======================
successful_extractions = sum(
    1 for result in results
    if result["Extracted_Text"] not in ["Failed to download PDF", "No text found", "Error extracting text"]
)
print(f"Processing complete!")
print(f"Used {PDF_LIBRARY} for PDF text extraction")
print(f"Successfully extracted text from {successful_extractions}/{len(results)} PDFs")
print(f"Results saved to {save_path}")

# Preview first successful extraction
for result in results:
    if len(result["Extracted_Text"]) > 100 and "Failed" not in result["Extracted_Text"]:
        preview = result["Extracted_Text"][:200].replace('\n', ' ')
        print(f"\nPreview of extracted text: {preview}...")
        break
