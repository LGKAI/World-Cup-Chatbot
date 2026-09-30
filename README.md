# World Cup Chatbot - Cháy cùng World Cup 2026

Xây dựng RAG Chatbot sử dụng thư viện LangChain và Ollama của Python.

## Cấu trúc thư mục

```text
Project_World-Cup-Chatbot/
│
├── faiss_index/                            # Vectorstore FAISS đã được lập chỉ mục sẵn
│   ├── index.faiss                         # File chỉ mục vector
│   └── index.pkl                           # Docstore chứa nội dung các đoạn văn bản (chunks)
│
├── papers/                                 # Tài liệu PDF tri thức gốc về World Cup
│   ├── world_cup_history_format.pdf        # Lịch sử và thể thức thi đấu World Cup
│   └── world_cup_stats_records.pdf         # Thống kê và kỷ lục các kỳ World Cup
│
├── chatbot.py                              # Mã nguồn chính chạy Chatbot RAG (Ensemble FAISS + BM25)
├── requirements.txt                        # Danh sách các thư viện Python phụ thuộc
└── README.md                               # Hướng dẫn cài đặt và thông tin dự án
```

**Chi tiết các thành phần:**
  - `faiss_index/`: Lưu trữ chỉ mục vector FAISS cục bộ. Khi khởi chạy, chương trình nạp trực tiếp vectorstore và trích xuất các text chunks từ docstore để khởi tạo bộ truy xuất BM25 mà không cần trích xuất lại file PDF.
  - `papers/`: Thư mục chứa các tài liệu PDF gốc cung cấp cơ sở tri thức cho chatbot.
  - `chatbot.py`: Kịch bản chính thực hiện kỹ thuật Hybrid Search (kết hợp FAISS và BM25 thông qua `EnsembleRetriever`), hỗ trợ ngữ cảnh hội thoại (`create_history_aware_retriever`) và trả lời câu hỏi bằng tiếng Việt qua mô hình Ollama.
  - `requirements.txt`: Liệt kê các thư viện cần thiết để chạy dự án.

## Hướng dẫn cài đặt

```text
1. Cài đặt Python, cài đặt môi trường ảo bằng Terminal:
    python -m venv venv
    .\venv\Scripts\activate

2. Tải các model AI bằng Terminal (yêu cầu đã cài đặt Ollama):
    ollama pull nomic-embed-text
    ollama pull qwen2.5:7b-instruct

3. Cài đặt các thư viện Python bằng Terminal:
    pip install -r requirements.txt

4. Khởi chạy Chatbot:
    python chatbot.py
```