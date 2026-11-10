"""Bước 1: đọc PDF, DOCX, TXT, XLSX và lưu văn bản thô thành JSON."""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from pypdf import PdfReader

from config import DATASET_DIR, STORAGE_DIR


def read_pdf(path: Path) -> list[dict]:
    results = []
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            results.append({"content": text, "location": f"page_{page_number}"})
    return results


def read_docx(path: Path) -> list[dict]:
    document = Document(path)
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [{"content": "\n".join(parts), "location": "document"}]


def read_txt(path: Path) -> list[dict]:
    return [{"content": path.read_text(encoding="utf-8"), "location": "document"}]


def read_excel(path: Path) -> list[dict]:
    results = []
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name).fillna("")
        # Đổi bảng thành text để embedding model hiểu được tên cột và giá trị.
        lines = [" | ".join(str(column) for column in frame.columns)]
        lines += [" | ".join(str(value) for value in row) for row in frame.values]
        results.append({"content": "\n".join(lines), "location": f"sheet:{sheet_name}"})
    return results


READERS = {".pdf": read_pdf, ".docx": read_docx, ".txt": read_txt, ".xlsx": read_excel}


def load_documents() -> list[dict]:
    documents = []
    for path in sorted(DATASET_DIR.iterdir()):
        reader = READERS.get(path.suffix.lower())
        if not reader:
            continue
        for item in reader(path):
            documents.append({
                "content": item["content"],
                "source": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "location": item["location"],
            })
        print(f"Đã đọc: {path.name}")
    return documents


def main() -> None:
    STORAGE_DIR.mkdir(exist_ok=True)
    documents = load_documents()
    output = STORAGE_DIR / "documents.json"
    output.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTổng số phần tài liệu: {len(documents)}")
    print(f"Đã lưu: {output}")


if __name__ == "__main__":
    main()

