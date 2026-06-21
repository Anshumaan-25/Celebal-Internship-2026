"""Generic ingestion for arbitrary PDFs / text files.

Used by the Streamlit demo (upload your own PDF) and any "bring your own
documents" experiment. The benchmark corpus is handled separately in
``dataset.py``.
"""
from __future__ import annotations

import io
from pathlib import Path


def read_pdf(path: str | Path) -> str:
    """Extract text from a PDF file on disk."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def read_pdf_bytes(data: bytes) -> str:
    """Extract text from in-memory PDF bytes (e.g. a Streamlit upload)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


def load_path(path: str | Path) -> str:
    """Read a single .pdf / .txt / .md file to raw text."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext in {".txt", ".md"}:
        return read_text(path)
    raise ValueError(f"Unsupported file type: {ext!r} (use .pdf, .txt or .md)")


def load_paths_as_documents(paths) -> list[dict]:
    """Load several files into pipeline-ready document records."""
    documents: list[dict] = []
    for path in paths:
        text = load_path(path)
        if text:
            documents.append(
                {"doc_id": Path(path).stem, "title": Path(path).name, "text": text}
            )
    return documents
