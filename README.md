# Chatbot RAG nội bộ — bản học tập

Project nhỏ để học đầy đủ các bước RAG bằng Python thuần. Dataset mô phỏng Công ty An Phát Digital và không chứa dữ liệu cá nhân thật. Project dùng FAISS CPU làm vector index và không dùng API LLM trả phí, LangChain hay giao diện web.

## Cấu trúc

```text
dataset/                    20 file PDF, DOCX, TXT, XLSX
notebooks/                  notebook cho bốn bước indexing
01_read_documents.py        đọc tài liệu
02_chunk_documents.py       chia chunk
03_create_embeddings.py     tạo vector
04_build_vector_store.py    lưu vector vào FAISS
05_search_documents.py      kiểm tra retrieval, chưa dùng LLM
06_chat_rag.py              retrieval + Ollama
config.py                   cấu hình
storage/                    kết quả trung gian
```

Notebook và script làm cùng một việc. Bạn có thể học bằng notebook ở bốn bước indexing, sau đó dùng script để chạy lại nhanh. Không cần chạy cả notebook lẫn script cho cùng một bước.

## 1. Chuẩn bị môi trường

Yêu cầu Python 3.10 trở lên.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install notebook
```

Ollama chỉ cần từ bước chatbot. Cài Ollama theo hệ điều hành, sau đó tải model nhẹ:

```bash
ollama pull qwen2.5:0.5b
```

Nếu máy chạy ổn và muốn câu trả lời tốt hơn, đổi `OLLAMA_MODEL` trong `config.py` thành `qwen2.5:1.5b` rồi tải model tương ứng.

## 2. Học bằng notebook

```bash
jupyter notebook
```

Mở và chạy theo thứ tự:

1. `notebooks/01_read_documents.ipynb`
2. `notebooks/02_chunk_documents.ipynb`
3. `notebooks/03_create_embeddings.ipynb`
4. `notebooks/04_build_vector_store.ipynb`

Mỗi notebook để trống output để bạn tự chạy và quan sát.

## 3. Hoặc chạy bằng script

```bash
python 01_read_documents.py
python 02_chunk_documents.py
python 03_create_embeddings.py
python 04_build_vector_store.py
```

Sau mỗi bước, kiểm tra thư mục `storage`:

- `documents.json`: văn bản được đọc từ file.
- `chunks.json`: các đoạn nhỏ cùng metadata nguồn.
- `embeddings.npy`: ma trận vector NumPy.
- `faiss.index`: vector index dùng để tìm kiếm.

## 4. Kiểm tra retrieval trước khi dùng LLM

```bash
python 05_search_documents.py
```

Câu hỏi gợi ý:

```text
Nhân viên được nghỉ phép bao nhiêu ngày?
Ticket P1 gói Business phản hồi trong bao lâu?
Đi công tác Hà Nội được thanh toán khách sạn tối đa bao nhiêu?
Mất laptop phải báo trong thời gian nào?
Chiết khấu 12% cần ai phê duyệt?
```

Hãy xem điểm similarity, nội dung chunk và nguồn. Nếu retrieval sai, LLM phía sau thường cũng trả lời sai.

## 5. Chạy chatbot RAG

Đảm bảo Ollama đang hoạt động, rồi chạy:

```bash
python 06_chat_rag.py
```

Gõ `exit` để thoát. Chatbot sẽ in câu trả lời và các nguồn được retrieval. Model 0.5B rất nhẹ nên đôi khi diễn đạt chưa tốt; mục tiêu chính của project là hiểu pipeline RAG.

## Luồng dữ liệu

```text
dataset/*
   -> documents.json
   -> chunks.json
   -> embeddings.npy
   -> FAISS index
   -> cosine similarity lấy top 4 chunk
   -> ghép context vào prompt
   -> Ollama tạo câu trả lời
```

## Thử thay đổi để học thêm

- Đổi `CHUNK_SIZE` từ 700 thành 400 hoặc 1.000 rồi so sánh kết quả.
- Đổi `CHUNK_OVERLAP` và xem thông tin ở ranh giới chunk.
- Đổi `TOP_K` từ 4 thành 2 hoặc 6.
- Thêm một tài liệu mới rồi chạy lại bốn bước indexing.
- Hỏi một câu không có trong tài liệu để kiểm tra khả năng từ chối.

