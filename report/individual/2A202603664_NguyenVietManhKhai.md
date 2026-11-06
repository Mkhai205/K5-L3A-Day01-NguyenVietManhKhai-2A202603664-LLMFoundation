# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603664
- **Họ và tên:** Nguyễn Viết Mạnh Khải
- **SEED:** 2517

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước (trang 1) | [x] |
| [`hand_calc_2.jpg`](hand_calc_2.jpg) — ảnh tính tay có các bước (trang 2) | [x] |

---

## 2. Kết quả chính

Bảng so sánh hiệu năng của 4 thế hệ AI trên cùng bài toán lọc SMS Spam:

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.744 | 0.853 | 0.857 | 0.189 ms | $0.000 |
| **2. BoW + MLP (deep learning)** | 0.973 | 0.878 | 0.923 | 0.919 | 9.030 ms | $0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.885 | 0.939 | 0.911 | 1.000 | 274.810 ms | $0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.987 | 0.902 | 0.943 | 0.974 | 81.590 ms | $0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.974 | 0.902 | 0.937 | 0.974 | 122.941 ms | $0.000 |
| **4a. LLM nhỏ zero-shot (gemini-3.5-flash-lite)** | 0.909 | 1.000 | 0.952 | 0.952 | 863.259 ms | $0.018 |
| **4b. LLM lớn zero-shot (gemini-3.8-flash)** | 0.947 | 0.900 | 0.923 | 0.923 | 4884.213 ms | $0.042 |
| **4c. LLM nhỏ few-shot** | 0.950 | 0.950 | 0.950 | 0.950 | 865.316 ms | $0.051 |

---

## 3. Dự đoán và thực tế

3 dự đoán lệch nhiều nhất và phân tích nguyên nhân:

1. **Dự đoán `llm_models` (So sánh LLM lớn vs LLM nhỏ):**
   - *Dự đoán:* Model lớn (`gemini-3.8-flash`) sẽ đạt F1 cao hơn model nhỏ (`gemini-3.5-flash-lite`) khoảng 0.02 - 0.05 nhờ khả năng suy luận ngữ cảnh sâu sắc hơn.
   - *Thực tế:* Model nhỏ lại đạt F1 cao hơn ($0.952$ so với $0.923$). Model lớn bị sai ở 2 tin spam thật dạng tường thuật/phàn nàn (`Bought one ringtone...`, `@A$NETWORKS...`).
   - *Nguyên nhân lệch:* Model lớn bị hiện tượng *"overthinking"* (suy nghĩ quá phức tạp), phân tích ngữ nghĩa quá mức các câu từ khiếu nại dịch vụ dẫn tới phán đoán nhầm là tin ham của người dùng. Model nhỏ giữ được ranh giới quyết định trực diện và chính xác hơn.

2. **Dự đoán `tradeoff` (Đường Pareto và kNN):**
   - *Dự đoán:* Các mô hình trên đường Pareto sẽ là BoW+LogReg (nhanh nhất), Encoder+LogReg (cân bằng) và LLM nhỏ (chất lượng cao nhất).
   - *Thực tế:* Mô hình `3a. Encoder + kNN` đạt F1 = 1.000 tuyệt đối trên tập SUB 40 tin (1 − F1 = 0%), biến chính nó thành điểm chặn cuối của Pareto và đẩy LLM nhỏ `4a` vào tập bị lấn át (*dominated*).
   - *Nguyên nhân lệch:* Do tập SUB chỉ có 40 tin nhắn (với 6 tin spam), kNN tìm được các láng giềng huấn luyện rất sát cho 6 tin này nên phân loại đúng 100%, tạo ra hiện tượng may mắn thống kê cục bộ trên tập con nhỏ.

3. **Dự đoán `hard_eval` (Khả năng chịu đựng trên 50 case khó):**
   - *Dự đoán:* Các mô hình BoW/MLP sẽ tụt khoảng 20-30%, vẫn giữ được accuracy ~60-70%.
   - *Thực tế:* BoW + LogReg tụt xuống đúng 50%, BoW + MLP tụt xuống 56% (gần như tương đương đoán ngẫu nhiên). Ở các nhóm đối kháng cụ thể như "Spam giọng tự nhiên" hay "Né từ khoá", BoW đoán đúng chỉ 0% - 17%.
   - *Nguyên nhân lệch:* BoW hoàn toàn phụ thuộc vào việc khớp chính xác chuỗi ký tự trong từ điển. Khi gặp thủ thuật né từ khoá (leetspeak) hoặc văn phong đời thường không có từ khóa cấm, cơ chế túi từ bị vô hiệu hóa 100%.

---

## 4. Phân tích 50 case khó

- **Model tụt nhiều nhất:** Hai mô hình BoW (`1. BoW + LogReg` và `2. BoW + MLP`) tụt sâu nhất, từ F1 test thường >0.85 - 0.92 tụt xuống chỉ còn 50% và 56% độ chính xác.
- **Nhóm case gây thảm họa cho BoW:**
  - Nhóm *Spam giọng tự nhiên*: BoW + LogReg đúng 0%, BoW + MLP đúng 10%.
  - Nhóm *Né từ khoá (Obfuscation)*: BoW + MLP đúng 0%, BoW + LogReg đúng 17%.
- **Hai case cụ thể và cơ chế gây sai:**
  1. **Case 15 (`natural_spam`):** `"Long time no see! I've started a little side b..."` (Nhãn thật: `spam`).
     - *Cơ chế sai:* Tin nhắn spam giả dạng bạn bè hỏi thăm thân mật, hoàn toàn không chứa từ khóa quảng cáo kinh điển ("win", "free", "cash", "prize"). BoW đếm tần suất từ thấy toàn từ ngữ giao tiếp thông thường nên đoán là `ham` (bỏ lọt spam).
  2. **Case 28 (`obfuscation`):** `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4r..."` (Nhãn thật: `spam`).
     - *Cơ chế sai:* Kẻ phát tán cố tình dùng số thay chữ cái (`0 -> o`, `4 -> a`, `1 -> i`). Do BoW chỉ tra cứu các từ nguyên bản trong từ điển `vocab`, toàn bộ các từ này trở thành *Out-of-Vocabulary (OOV)* và bị loại bỏ. Vector đặc trưng trở thành toàn số 0, mô hình không phát hiện được dấu hiệu spam nào.
- **Hiệu quả của Prompt phòng thủ:** Hàng `4d` với thẻ bọc `<sms>...</sms>` và chỉ thị phân định dữ liệu/lệnh đã đạt **100% độ chính xác tuyệt đối (50/50 case)**, không bị lừa bởi bất kỳ đòn Prompt Injection nào.

---

## 5. Thử thách cá nhân

- **Đề bài (Thử thách #4):** Phân tích mô hình `kNN` trên không gian embedding với các giá trị $k \in \{1, 5, 25\}$.
- **Cách làm:** Gọi hàm `knn_predict(E_tr, y_tr, E_te, k=k_val)` trên tập test với lần lượt $k = 1, 5, 25$ và tính `metrics(y_te, preds)`.
- **Kết quả thực tế:**
  - $k = 1$: Accuracy = 0.984, Precision = 0.880, Recall = 0.890, **F1 = 0.885**
  - $k = 5$: Accuracy = 0.988, Precision = 0.885, Recall = 0.939, **F1 = 0.911** (Tốt nhất)
  - $k = 25$: Accuracy = 0.982, Precision = 0.942, Recall = 0.793, **F1 = 0.861**
- **Giải thích cơ chế:**
  - Khi $k=1$: Quyết định phụ thuộc vào duy nhất 1 điểm gần nhất. Mô hình rất nhạy cảm với nhiễu và các điểm ngoại lai (outliers) gần ranh giới quyết định $\rightarrow$ Overfitting cục bộ, F1 thấp.
  - Khi $k=5$: Gom phiếu bầu của 5 điểm lân cận giúp triệt tiêu nhiễu tốt trong khi vẫn giữ được cấu trúc phân vùng quyết định địa phương $\rightarrow$ Cân bằng tối ưu giữa Precision và Recall.
  - Khi $k=25$: Lấy láng giềng quá rộng. Do tập dữ liệu mất cân bằng nặng (Ham chiếm ~87%, Spam chỉ ~13%), vùng láng giềng lớn bị áp đảo bởi các điểm Ham, kéo Recall của lớp Spam giảm mạnh từ 0.939 xuống 0.793 $\rightarrow$ Underfitting.

---

## 6. Phương án cho sản phẩm thật

- **Bài toán đặt ra:** Hệ thống viễn thông xử lý **10 triệu SMS / ngày**, yêu cầu SLA độ trễ $p95 < 50\text{ ms/tin}$.
- **Phương án lựa chọn:** **Mô hình kết hợp 2 tầng (Cascading Pipeline)** với nòng cốt là **`2. BoW + MLP`**:
  - *Tầng 1 (95% lưu lượng):* Chạy mô hình **`2. BoW + MLP`** trực tiếp tại gateway:
    - *Số liệu chứng minh:* $p95 = 9.03\text{ ms}$ (thoải mái nằm sâu dưới SLA 50 ms), thông lượng đạt $5,804\text{ tin/s}$ trên 1 core, chi phí API = $0 USD$. Xử lý 10 triệu tin chỉ mất vài chục phút tính toán cục bộ.
    - Với các tin nhắn có xác suất dự đoán rõ ràng ($P(\text{spam}) > 0.9$ hoặc $P(\text{spam}) < 0.1$), đưa ra quyết định ngay lập tức.
  - *Tầng 2 (5% ca nghi ngờ):* Với các tin nhắn nằm ở vùng bất định ($0.1 \le P \le 0.9$), áp dụng chuẩn hóa ký tự leetspeak và đẩy qua mô hình **`3b. Encoder (MiniLM) + LogReg`** được tối ưu qua ONNX/TensorRT (giúp giảm $p95$ từ 81 ms xuống < 20 ms).
- **Đánh đổi chấp nhận:**
  - Không sử dụng LLM cho toàn bộ 10 triệu tin: Vì với LLM nhỏ ($p95 = 863\text{ ms}$, chi phí $0.018$ / 1k tin), 10 triệu tin/ngày sẽ tiêu tốn **$180 USD / ngày** ($~5,400 USD / tháng) và vi phạm hoàn toàn SLA 50 ms của nhà mạng.

---

## 7. AI log và điều đã học

- **AI Log:**
  1. *Lần 1 (Phần 4 - Gọi LLM API & Prompt):* Hỏi AI về cú pháp cấu trúc `build_classification_prompt` và hàm bóc tách `parse_label`. Nhận code mẫu bóc tách bằng regex. Tôi đã sửa lại để kiểm soát chặt trường hợp model trả lời chuỗi chứa cả "not spam" nhằm tránh False Positive.
  2. *Lần 2 (Phần 5 - Prompt phòng thủ):* Hỏi cách thiết kế thẻ XML `<sms>` để chặn prompt injection. Tôi đã kiểm tra kỹ việc làm sạch ký tự thoát thẻ `text.replace("<sms>", "")` để đảm bảo attacker không thể escape prompt.
- **Điều hiểu ra sau bài lab:**
  1. *Độ phức tạp của mô hình không đồng nghĩa với hiệu quả phân loại:* LLM lớn (Gemini 3.8 Flash) hoàn toàn có thể thua LLM nhỏ (Gemini 3.5 Flash Lite) do hiện tượng overthinking ở các tác vụ phân loại nhãn đóng.
  2. *Độ trễ và chi phí là rào cản quyết định trên production:* Mô hình đạt F1 cao nhất chưa chắc đã là mô hình được chọn. Sự đánh đổi giữa F1, $p95$ và chi phí vận hành (đường Pareto) mới là tiêu chuẩn thiết kế kiến trúc thực tế.
  3. *Tập test IID luôn quá lạc quan:* Điểm số F1 > 0.90 trên tập test thông thường chỉ là ảo tưởng an toàn nếu chưa vượt qua các bài stress test đối kháng (adversarial test).
