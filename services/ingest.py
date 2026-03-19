from __future__ import annotations
from pathlib import Path

from docx import Document
from pypdf import PdfReader


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_docx(file_bytes: bytes) -> str:
    tmp = Path("_tmp_upload.docx")
    tmp.write_bytes(file_bytes)
    doc = Document(str(tmp))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return text


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import io

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        if t.strip():
            pages.append(t)
    return "\n".join(pages)


def extract_text(file_name: str, file_bytes: bytes) -> str:
    ext = Path(file_name).suffix.lower()
    if ext == ".txt":
        return extract_text_from_txt(file_bytes)
    if ext == ".docx":
        return extract_text_from_docx(file_bytes)
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    raise ValueError(f"Unsupported file type: {ext}")