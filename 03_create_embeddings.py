"""Bước 3: biến nội dung từng chunk thành vector và lưu dưới dạng NumPy."""
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, STORAGE_DIR


def main() -> None:
    chunks = json.loads((STORAGE_DIR / "chunks.json").read_text(encoding="utf-8"))
    texts = [chunk["content"] for chunk in chunks]
    print(f"Đang nạp embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    output_path = STORAGE_DIR / "embeddings.npy"
    np.save(output_path, embeddings)
    print(f"Số chunk: {len(chunks)}")
    print(f"Kích thước ma trận: {embeddings.shape}")
    print(f"10 số đầu vector đầu tiên: {embeddings[0][:10]}")
    print(f"Đã lưu: {output_path}")


if __name__ == "__main__":
    main()

