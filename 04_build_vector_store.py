"""Bước 4: đưa các vector đã tạo vào FAISS và lưu index xuống ổ đĩa."""
import json

import faiss
import numpy as np

from config import STORAGE_DIR


def main() -> None:
    chunks_path = STORAGE_DIR / "chunks.json"
    embeddings_path = STORAGE_DIR / "embeddings.npy"

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path).astype("float32")

    if len(chunks) != len(embeddings):
        raise ValueError("Số chunk và số vector không bằng nhau")

    # Các vector đã được normalize ở bước 3. IndexFlatIP dùng tích vô hướng,
    # vì vậy điểm trả về tương đương cosine similarity.
    vector_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(vector_dimension)
    index.add(embeddings)

    output_path = STORAGE_DIR / "faiss.index"
    faiss.write_index(index, str(output_path))

    print(f"Loại index: {type(index).__name__}")
    print(f"Số chiều vector: {vector_dimension}")
    print(f"Số vector trong index: {index.ntotal}")
    print(f"Đã lưu: {output_path}")


if __name__ == "__main__":
    main()

