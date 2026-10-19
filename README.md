# World Cup Chatbot

Ứng dụng RAG Chatbot thông minh chuyên hỗ trợ giải đáp mọi thông tin, lịch sử, thể thức và số liệu thống kê về World Cup. Hệ thống được xây dựng hoàn toàn cục bộ (local) kết hợp thư viện **LangChain** và **Ollama**.

> **📌 Phân loại RAG:** Thuộc kiến trúc **Advanced RAG (RAG Nâng cao)**, là sự kết hợp giữa:
> - **Hybrid Search RAG (RAG Tìm kiếm kết hợp):** Khắc phục nhược điểm của Naive RAG bằng cách kết hợp truy xuất ngữ nghĩa dense vector (**FAISS**) và truy xuất từ khóa chính xác sparse (**BM25**).
> - **Conversational / History-Aware RAG (RAG Đàm thoại theo ngữ cảnh):** Tự động phân tích lịch sử hội thoại để viết lại câu hỏi (Query Rewriting) trước khi truy xuất dữ liệu.

---

## 🌟 Tính năng nổi bật

- **Tìm kiếm kết hợp (Hybrid Search):** Kết hợp tìm kiếm ngữ nghĩa vector qua **FAISS** (Dense Retrieval) và tìm kiếm từ khóa chính xác qua **BM25** (Sparse Retrieval) với trọng số tối ưu.
- **Ghi nhớ ngữ cảnh đàm thoại (History-Aware Retriever):** Tự động hiểu và viết lại câu hỏi dựa trên lịch sử trao đổi trước đó.
- **Hoạt động hoàn toàn Offline / Local:** Bảo mật thông tin, không phụ thuộc API trả phí ngoài, vận hành thông qua các mô hình AI mã nguồn mở trên Ollama.
- **Tối ưu tiếng Việt:** LLM trả lời tự nhiên, chính xác, bám sát tài liệu nguồn.

---

## 🏗️ Kiến trúc & Cách thức hoạt động

Hệ thống hoạt động theo kiến trúc RAG nâng cao kết hợp 2 quy trình chính: **Lập chỉ mục (Indexing Pipeline)** và **Truy xuất - Phản hồi (Query & Retrieval Pipeline)**. Chi tiết:

1. **Thu thập & Cắt đoạn tài liệu (Document Ingestion & Chunking):**
   - Đọc các file PDF trong thư mục `papers/` bằng `UnstructuredFileLoader` đa luồng.
   - Sử dụng `RecursiveCharacterTextSplitter` với kích thước đoạn `1200` ký tự và độ gối đầu `200` ký tự (`overlap`). Hệ thống phân tách thông minh theo cấu trúc tiêu đề, đoạn văn để bảo toàn trọn vẹn ngữ nghĩa từng sự kiện bóng đá.

2. **Nhúng Vector & Lưu trữ chỉ mục (Embedding & FAISS Vectorstore):**
   - Các đoạn văn bản được mô hình `nomic-embed-text` chuyển hóa thành vector đặc trưng.
   - Lưu trữ vào cơ sở dữ liệu vector **FAISS** theo khoảng cách Cosine (`DistanceStrategy.COSINE`) và lưu cục bộ tại `faiss_index/`. Khi ứng dụng chạy ở các lần tiếp theo, chỉ mục sẽ được nạp trực tiếp chỉ trong tích tắc mà không cần trích xuất lại file PDF.

3. **Truy xuất lai (Hybrid Search / Ensemble Retrieval):**
   - **Dense Retrieval (FAISS - 40% trọng số):** Tìm kiếm theo độ tương đồng ngữ nghĩa vector, nhận diện được câu hỏi diễn đạt khác từ nhưng cùng ý nghĩa.
   - **Sparse Retrieval (BM25 - 60% trọng số):** Tìm kiếm từ khóa chính xác dựa trên tần suất từ, đặc biệt hiệu quả với tên riêng cầu thủ, đội bóng, năm tổ chức, tỷ số và các kỷ lục chi tiết.
   - `EnsembleRetriever` kết hợp kết quả từ cả 2 bộ truy xuất để trả về các đoạn ngữ cảnh phù hợp nhất.

4. **Nhận diện ngữ cảnh đàm thoại (History-Aware Retriever):**
   - Khi người dùng hỏi nối tiếp các câu hỏi có đại từ thay thế (ví dụ: *"Kỳ đó tổ chức ở đâu?", "Anh ấy ghi được bao nhiêu bàn?"*), mô hình LLM sẽ tham chiếu `chat_history` và tự động tái cấu trúc thành một câu hỏi độc lập trước khi thực hiện tìm kiếm.

5. **Sinh câu trả lời bám sát ngữ cảnh (Grounded Generation):**
   - Đưa ngữ cảnh tìm được vào mô hình `qwen2.5:7b-instruct` kèm chỉ dẫn hệ thống nghiêm ngặt:
     - Trả lời trực tiếp vào trọng tâm, có kèm số liệu/bối cảnh mở rộng.
     - Tuyệt đối trung thực với tài liệu, không bịa đặt thông tin ngoài. Nếu tài liệu không đề cập, chatbot sẽ phản hồi rõ: *"Tôi không tìm thấy thông tin này trong tài liệu."*
     - Định dạng tiếng Việt tự nhiên, đàm thoại mượt mà.

---

## 📁 Cấu trúc thư mục

```text
Project_World-Cup-Chatbot/
│
├── faiss_index/                            # Vectorstore FAISS đã được lập chỉ mục cục bộ
│   ├── index.faiss                         # File chỉ mục vector
│   └── index.pkl                           # Docstore chứa nội dung các đoạn văn bản (chunks)
│
├── papers/                                 # Thư mục tài liệu PDF tri thức gốc
│   ├── lich_su_the_thuc.pdf                # Lịch sử và thể thức thi đấu World Cup
│   └── thong_ke_ki_luc.pdf                 # Thống kê và kỷ lục các kỳ World Cup
│
├── chatbot.py                              # Mã nguồn chính chạy Chatbot RAG
├── requirements.txt                        # Danh sách các thư viện Python phụ thuộc
└── README.md                               # Hướng dẫn cài đặt và tài liệu dự án
```

---

## 📋 Yêu cầu hệ thống

1. **Python**: Phiên bản >= 3.10 (Khuyên dùng Python 3.11 hoặc 3.12).
2. **Ollama**: Đã cài đặt [Ollama](https://ollama.com/) trên hệ thống.

---

## 🚀 Hướng dẫn cài đặt & Khởi chạy

### Bước 1: Khởi động Ollama và tải các Model AI

Đảm bảo dịch vụ Ollama đang chạy trên máy tính của bạn, sau đó mở Terminal và chạy:

```powershell
# Tải mô hình Embedding (để chuyển văn bản thành vector)
ollama pull nomic-embed-text

# Tải mô hình LLM chính (để xử lý ngôn ngữ và trả lời câu hỏi)
ollama pull qwen2.5:7b-instruct
```

> **Lưu ý:** Hãy luôn giữ ứng dụng Ollama chạy ngầm (hoặc chạy lệnh `ollama serve`) trước khi khởi chạy chatbot.

---

### Bước 2: Chuẩn bị môi trường Python

Mở Terminal tại thư mục dự án `Project_World-Cup-Chatbot`:

**Cách 1: Sử dụng môi trường ảo (Khuyên dùng để tránh xung đột thư viện)**
```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (trên Windows PowerShell)
.\venv\Scripts\activate

# Nếu gặp lỗi không cho chạy script trên PowerShell, cấp quyền tạm thời:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Cách 2: Sử dụng Python trực tiếp trên hệ thống**  
Bạn có thể bỏ qua bước kích hoạt môi trường ảo và sử dụng Python sẵn có.

---

### Bước 3: Cài đặt các thư viện phụ thuộc

Cài đặt các gói thư viện cần thiết qua `pip`:

```powershell
pip install -r requirements.txt
```

---

### Bước 4: Khởi chạy Chatbot

Khởi động giao diện trò chuyện trên Terminal:

```powershell
python chatbot.py
```

- Khi khởi chạy lần đầu hoặc khi chưa có thư mục `faiss_index/`, hệ thống sẽ tự động đọc các tài liệu PDF trong thư mục `papers/`, cắt nhỏ văn bản và tạo cơ sở dữ liệu vector.
- Khi chỉ mục đã tồn tại, chatbot sẽ nạp trực tiếp chỉ mục trong vài giây.
- Gõ câu hỏi của bạn và nhấn `Enter`. Gõ `exit` để thoát.

---

## 🔄 Cách cập nhật / Thay đổi tài liệu PDF mới

Khi bạn muốn thêm, xóa hoặc thay đổi các file PDF trong thư mục `papers/`:

1. Sao chép các file PDF mới vào thư mục `papers/` (và xóa các file cũ không dùng).
2. **Xóa thư mục `faiss_index/`** trong dự án:
   ```powershell
   Remove-Item -Recurse -Force faiss_index
   ```
3. Chạy lại lệnh:
   ```powershell
   python chatbot.py
   ```
   Hệ thống sẽ tự động quét lại toàn bộ tài liệu mới trong `papers/` và tái tạo cơ sở dữ liệu tri thức mới.