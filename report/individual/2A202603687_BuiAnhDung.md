# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603687
- **Họ và tên:** Bùi Anh Dũng
- **SEED:** 9468

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb): notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json): do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg): ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách | F1 (SUB 40) | F1 (test) | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.919 | 0.880 | 0.10 | 0 |
| 2. BoW + MLP | 1.000 | 0.939 | 2.50 | 0 |
| 3a. Encoder + kNN | 0.950 | 0.872 | 68.15 | 0 |
| 3b. Encoder + LogReg | 0.974 | 0.955 | 41.34 | 0 |
| 3c. Encoder + MLP head | 0.974 | 0.960 | 58.57 | 0 |
| 4a. LLM nhỏ zero-shot (Groq gpt-oss-20b) | 0.837 | - | 1769.36 | 0 (gói miễn phí) |
| 4b. LLM lớn zero-shot (Groq gpt-oss-120b) | 0.976 | - | 412.34 | 0 (gói miễn phí) |
| 4c. LLM nhỏ few-shot | 0.974 | - | 1438.07 | 0 (gói miễn phí) |

Các hàng LLM chỉ đo trên tập SUB (40 tin) nên cột F1 (test) để trống; số lấy từ `lab1_report.json`.

## 3. Dự đoán và thực tế

- **lr = 50 (ml_lr):** mình đoán loss dao động và không giảm; thực tế loss chỉ vọt lên ở vòng đầu rồi hội tụ nhanh, loss cuối thấp nhất (0.006), vì gradient bị chặn trong [-1, 1].
- **Few-shot (llm_fewshot):** mình đoán F1 chỉ tăng nhẹ; thực tế F1 model nhỏ tăng từ 0.837 lên 0.974, gần bằng model lớn.
- **MLP so với LogReg (dl_curve):** mình đoán ngang nhau; thực tế 0.939 so với 0.88.

## 4. Phân tích 50 case khó

BoW + MLP tụt nhiều nhất: F1 từ 1.000 xuống 0.286; BoW + LogReg từ 0.919 xuống 0.286. LLM không tụt (4a: 0.903, 4d: 0.935). Với BoW, nhóm khó nhất là spam giọng tự nhiên (0%), thao túng (12%), né từ khoá (17% hoặc 0%), vì BoW chỉ cộng điểm từ rời trong từ điển và từ như C0ngr4tulat10ns không có trong từ điển. Case #47 ("lol I just got a text saying I won £1000...", thật là ham): LLM dùng prompt phòng thủ (4d) vẫn đoán spam vì bám dấu hiệu bề mặt như "won £1000", "0906 number".

## 5. Thử thách cá nhân

Đề #1 (đổi ngưỡng quyết định của BoW + LogReg): ngưỡng 0.2 cho precision 0.881, recall 0.932, F1 0.906; ngưỡng 0.5 cho precision 1.0, recall 0.786, F1 0.88; ngưỡng 0.8 cho precision 1.0, recall 0.563, F1 0.72. Ngân hàng sợ chặn nhầm OTP ưu tiên precision nên không chọn ngưỡng thấp; mình chọn 0.5 vì precision đã 1.0 mà recall còn 0.786, và nâng lên 0.6 đến 0.8 nếu muốn an toàn hơn.

## 6. Phương án cho sản phẩm thật

Yêu cầu p95 dưới 50 ms cho 10 triệu SMS/ngày: chọn BoW + MLP làm lớp chính (p95 2.5 ms, F1 test 0.939, thông lượng ~12.700 tin/s). Encoder + LogReg có F1 test cao hơn (0.955) và p95 41 ms vẫn dưới SLA nhưng sát ngưỡng trên CPU, nên dùng làm ý kiến thứ hai trong router thay vì lớp chính. Router: tin lưỡng lự (p từ 0.3 đến 0.7) hoặc BoW và Encoder bất đồng thì chuyển cho LLM. LLM đơn lẻ không đạt SLA (p95 0.4 đến 1.8 giây) và nếu tính giá gpt-6-luna thì khoảng 105 USD/ngày (~3.150 USD/tháng).

## 7. AI log và điều đã học

AI log: hỏi AI giải thích khái niệm, cú pháp và nhận xét lỗi trong code của mình (Phần 0 đến 2); AI viết giúp 5 dòng `train_mlp`, Phần 3 (`cosine_similarity`, `top_k_similar`, `knn_predict`) và Phần 4 (`call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch`); AI soạn nháp một số câu trả lời, mình chỉnh theo số liệu của mình; AI gợi ý thêm `time.sleep` và đổi sang Groq khi gặp lỗi 429.

Điều hiểu ra hôm nay: phải đo bằng thực nghiệm chứ không chỉ suy luận; F1 trên test thường rất lạc quan so với 50 case khó; và vài ví dụ trong prompt (few-shot) có thể đưa model nhỏ ngang model lớn.
