"""Bước 2: làm sạch và chia văn bản thành các đoạn nhỏ có overlap."""
import json
import re

from config import CHUNK_OVERLAP, CHUNK_SIZE, STORAGE_DIR


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chia theo ký tự; cố gắng dừng ở cuối dòng hoặc cuối câu."""
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP phải nhỏ hơn CHUNK_SIZE")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            candidates = [text.rfind("\n", start, end), text.rfind(". ", start, end)]
            natural_end = max(candidates)
            if natural_end > start + chunk_size // 2:
                end = natural_end + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main() -> None:
    input_path = STORAGE_DIR / "documents.json"
    documents = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = []
    for document in documents:
        text = clean_text(document["content"])
        for part_number, content in enumerate(split_text(text, CHUNK_SIZE, CHUNK_OVERLAP), start=1):
            chunks.append({
                "chunk_id": len(chunks),
                "content": content,
                "source": document["source"],
                "file_type": document["file_type"],
                "location": document["location"],
                "part": part_number,
            })
    output_path = STORAGE_DIR / "chunks.json"
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    average = sum(len(x["content"]) for x in chunks) / max(len(chunks), 1)
    print(f"Số phần tài liệu đầu vào: {len(documents)}")
    print(f"Số chunk: {len(chunks)}")
    print(f"Độ dài trung bình: {average:.0f} ký tự")
    print(f"Đã lưu: {output_path}")


if __name__ == "__main__":
    main()

