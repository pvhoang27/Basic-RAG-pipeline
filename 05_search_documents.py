"""Bước 5: tìm các chunk gần câu hỏi nhất bằng FAISS, chưa dùng LLM."""
import json

import faiss
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, STORAGE_DIR, TOP_K


def load_data():
    chunks = json.loads((STORAGE_DIR / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(STORAGE_DIR / "faiss.index"))
    if index.ntotal != len(chunks):
        raise ValueError("Số vector trong FAISS không khớp số chunk")
    return chunks, index


def search(question, model, chunks, index, top_k=TOP_K):
    question_vector = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(question_vector, min(top_k, index.ntotal))
    return [(chunks[i], float(score)) for i, score in zip(indices[0], scores[0]) if i >= 0]


def main() -> None:
    chunks, index = load_data()
    model = SentenceTransformer(EMBEDDING_MODEL)
    while True:
        question = input("\nNhập câu hỏi (gõ exit để thoát): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        for rank, (chunk, score) in enumerate(search(question, model, chunks, index), start=1):
            print(f"\n--- Kết quả {rank} | Điểm: {score:.4f} ---")
            print(f"Nguồn: {chunk['source']} | {chunk['location']} | phần {chunk['part']}")
            print(chunk["content"])


if __name__ == "__main__":
    main()

