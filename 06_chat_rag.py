"""Bước 6: retrieval từ FAISS + prompt + LLM local qua Ollama."""
import json

import faiss
import requests
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, OLLAMA_MODEL, OLLAMA_URL, STORAGE_DIR, TOP_K


def retrieve(question, model, chunks, index):
    question_vector = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(question_vector, min(TOP_K, index.ntotal))
    return [
        {**chunks[i], "score": float(score)}
        for i, score in zip(indices[0], scores[0])
        if i >= 0
    ]


def build_prompt(question, results):
    context_parts = []
    for number, item in enumerate(results, start=1):
        context_parts.append(
            f"[Tài liệu {number}: {item['source']} - {item['location']}]\n{item['content']}"
        )
    context = "\n\n".join(context_parts)
    return f"""Bạn là trợ lý nội bộ của Công ty An Phát Digital.
Chỉ trả lời dựa trên tài liệu tham khảo bên dưới.
Không tự thêm thông tin. Nếu tài liệu không đủ, hãy nói rõ:
\"Tôi không tìm thấy thông tin này trong tài liệu nội bộ.\"
Trả lời ngắn gọn, rõ ràng bằng tiếng Việt.

TÀI LIỆU THAM KHẢO:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""


def ask_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


def main() -> None:
    chunks = json.loads((STORAGE_DIR / "chunks.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(STORAGE_DIR / "faiss.index"))
    if index.ntotal != len(chunks):
        raise ValueError("Số vector trong FAISS không khớp số chunk")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Chatbot RAG đang dùng {OLLAMA_MODEL}. Gõ exit để thoát.")
    while True:
        question = input("\nBạn: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        results = retrieve(question, model, chunks, index)
        try:
            answer = ask_ollama(build_prompt(question, results))
        except requests.RequestException as error:
            print(f"Không kết nối được Ollama: {error}")
            continue
        print(f"\nTrợ lý: {answer.strip()}")
        print("\nNguồn được truy xuất:")
        for item in results:
            print(f"- {item['source']} | {item['location']} | điểm {item['score']:.4f}")


if __name__ == "__main__":
    main()

