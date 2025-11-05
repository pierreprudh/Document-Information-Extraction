import os
from pathlib import Path
from typing import Annotated

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination

from dotenv import load_dotenv
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

from io import BytesIO
from functools import lru_cache
import numpy as np
import cv2

from mistralai import Mistral

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
APP_ROOT = Path(__file__).resolve().parents[2]

api_key_mistral = os.getenv("API_KEY_MISTRAL")


def tesseract_ocr(
    file_path: Annotated[str, "Path to the image or PDF file to analyze"]
) -> Annotated[str, "Extracted text from the file or error message"]:
    """Extract text from image or PDF using Tesseract OCR.

    Args:
        file_path: Path to the image or PDF file to process

    Returns:
        Extracted text or error message
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: file '{file_path}' does not exist."

        if not path.is_file():
            return f"Error: '{file_path}' is not a valid file."

        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.pdf', '.txt'}
        if path.suffix.lower() not in supported_formats:
            return f"Error: format '{path.suffix}' not supported. Accepted formats: {', '.join(supported_formats - {'.txt'})}"

        config = r'--oem 3 --psm 6'
        extracted_text = ""

        if path.suffix.lower() == '.txt':
            print(f"Opening text file: {file_path}")
            with open(path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
        elif path.suffix.lower() == '.pdf':
            print(f"Converting PDF pages to images: {file_path}")
            images = convert_from_path(file_path)
            for i, img in enumerate(images, 1):
                print(f"Processing page {i}")
                text = pytesseract.image_to_string(img, lang='eng+fra', config=config)
                extracted_text += f"\n--- Page {i} ---\n{text.strip()}\n"
        else:
            print(f"Opening image: {file_path}")
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='eng+fra', config=config)
            extracted_text = text

        if not extracted_text.strip():
            return "No text detected. Please check that the file contains readable text."

        cleaned_text = '\n'.join(line.strip() for line in extracted_text.splitlines() if line.strip())
        return f"Text extracted successfully:\n\n{cleaned_text}"

    except pytesseract.TesseractNotFoundError:
        return "Error: Tesseract is not installed or not found in PATH."
    except Exception as e:
        return f"OCR Error: {str(e)}"

def extract_info_from_pdf_with_mistral_ocr(path, api_key=api_key_mistral):
    """
    Extract text from a PDF or image using Mistral OCR.
    - path: path to PDF or image file
    - api_key: Mistral API key (defaults to env var API_KEY_MISTRAL or MISTRAL_API_KEY)
    """
    if not api_key:
        raise ValueError("Mistral API key not set. Define MISTRAL_API_KEY or API_KEY_MISTRAL, or pass api_key explicitly.")
    client = Mistral(api_key=api_key)

    file_path = Path(path)
    assert file_path.exists(), "Le fichier spécifié n'existe pas."

    uploaded_file = client.files.upload(file={
        "file_name": file_path.name,
        "content": file_path.read_bytes()
    }, purpose="ocr")

    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        },
    )

    pages = getattr(ocr_response, "pages", []) or []
    if not pages:
        print("[WARN] OCR returned no pages. Check file readability or permissions.")
        return ""

    extracted_text = "\n".join(getattr(p, "markdown", "") for p in pages)


    return extracted_text


def tesseract_ocr_image(image_data, config: str = r"--oem 3 --psm 6") -> str:
    """OCR with Tesseract from in-memory image data (bytes, PIL, or np.ndarray)."""
    try:
        if isinstance(image_data, bytes):
            image = Image.open(BytesIO(image_data))
        elif hasattr(image_data, "save"):
            image = image_data
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            return "Tesseract Error: Unsupported image format"
        text = pytesseract.image_to_string(image, lang="eng+fra", config=config)
        return text
    except Exception as e:
        hint = ""
        if "numpy.dtype size changed" in str(e):
            hint = " Possible fix: align your NumPy/Pandas versions (e.g., reinstall both) or uninstall pandas if you don't need DataFrame outputs for Tesseract."
        return f"Tesseract Error: {str(e)}{hint}"
@lru_cache(maxsize=1)
def _get_easyocr_reader():
    import easyocr  # Lazy import
    return easyocr.Reader(["en", "fr"])  # cached
def easyocr_process(image_data) -> str:
    """OCR with EasyOCR from in-memory image data (bytes, PIL, or np.ndarray)."""
    try:
        reader = _get_easyocr_reader()
        if isinstance(image_data, bytes):
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif hasattr(image_data, "save"):
            image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        elif isinstance(image_data, np.ndarray):
            image = image_data if image_data.ndim == 3 else cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
        else:
            return "EasyOCR Error: Unsupported image format"
        results = reader.readtext(image)
        text = "\n".join([result[1] for result in results])
        return text
    except Exception as e:
        return f"EasyOCR Error: {str(e)}"
@lru_cache(maxsize=1)
def _get_paddle_ocr():
    from paddleocr import PaddleOCR  # Lazy import
    return PaddleOCR(use_angle_cls=True, lang="fr")
def paddleocr_process(image_data) -> str:
    """OCR with PaddleOCR from in-memory image data (bytes, PIL, or np.ndarray)."""
    try:
        ocr = _get_paddle_ocr()
        if isinstance(image_data, bytes):
            image_array = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        elif hasattr(image_data, "save"):
            image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
        elif isinstance(image_data, np.ndarray):
            image = image_data if image_data.ndim == 3 else cv2.cvtColor(image_data, cv2.COLOR_GRAY2BGR)
        else:
            return "PaddleOCR Error: Unsupported image format"
        results = ocr.ocr(image, cls=True)
        text = "\n".join([line[1][0] for page in results for line in page if line])
        return text
    except Exception as e:
        return f"PaddleOCR Error: {str(e)}"

def azure_ocr(image_data) -> str:
    """OCR with Azure Document Intelligence from in-memory image data (bytes, PIL, or np.ndarray)."""
    try:
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.ai.documentintelligence.models import AnalyzeResult
        except ImportError:
            return (
                "Azure Error: azure-ai-documentintelligence is not installed or incompatible. "
                "Install it with: pip install --upgrade azure-ai-documentintelligence"
            )
        endpoint = os.getenv("AZURE_ENDPOINT", "")
        azure_key = os.getenv("AZURE_KEY", "")
        if not endpoint or not azure_key:
            return "Azure Error: AZURE_ENDPOINT and AZURE_KEY environment variables are required"
        credential = AzureKeyCredential(azure_key)
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)
        content_type = "image/jpeg"
        try:
            if isinstance(image_data, bytes):
                pil_img = Image.open(BytesIO(image_data))
            elif hasattr(image_data, "save"):
                pil_img = image_data
            elif isinstance(image_data, np.ndarray):
                pil_img = Image.fromarray(image_data)
            else:
                return "Azure Error: Unsupported image format"
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")
            img_buffer = BytesIO()
            pil_img.save(img_buffer, format="JPEG", quality=95, optimize=True)
            document_bytes = img_buffer.getvalue()
        except Exception:
            return "Azure Error: Failed to normalize image bytes (possibly corrupted input)."
        if len(document_bytes) > 20 * 1024 * 1024:
            return "Azure Error: Image exceeds 20 MB after normalization. Please downscale and try again."
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=document_bytes,
            content_type=content_type,
        )
        result: AnalyzeResult = poller.result()
        extracted_text = ""
        if getattr(result, "content", None):
            extracted_text = result.content
        else:
            if getattr(result, "pages", None):
                for page in result.pages:
                    if getattr(page, "lines", None):
                        for line in page.lines:
                            extracted_text += line.content + "\n"
        return extracted_text.strip() if extracted_text else "No text detected"
    except Exception as e:
        return f"Azure Error: {str(e)}"
