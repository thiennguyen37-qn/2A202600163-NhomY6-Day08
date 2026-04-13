# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyễn Hồ Bảo Thiên 
**Vai trò trong nhóm:** Eval Owner
**Ngày nộp:** 13/04/2026
**Độ dài yêu cầu:** 500–800 từ

---

## 1. Tôi đã làm gì trong lab này? (100-150 từ)

> Mô tả cụ thể phần bạn đóng góp vào pipeline:
> - Sprint nào bạn chủ yếu làm?
> - Cụ thể bạn implement hoặc quyết định điều gì?
> - Công việc của bạn kết nối với phần của người khác như thế nào?

Trong lab này, tôi đảm nhận phần **Sprint 4: Evaluation & Scorecard** thông qua việc xây dựng và hoàn thiện file `eval.py`. Cụ thể, tôi đã implement logic cho các hàm đánh giá gồm: score_faithfulness, score_answer_relevance, score_context_recall, score_completeness. Với `score_context_recall`, tôi đã tối ưu hóa việc so sánh chuỗi bằng thư viện `os.path` để đối chiếu linh hoạt các đường dẫn file (tránh lỗi khi format path khác nhau). Với `score_completeness`, tôi thiết lập prompt để sử dụng phương pháp **LLM-as-Judge**, gọi API để AI tự động đối chiếu câu trả lời với đáp án chuẩn. Công việc của tôi đóng vai trò là "bộ lọc" cuối cùng, kết nối với Sprint 2 và 3 của các thành viên khác, chạy qua bộ test, và kiểm tra các variant có thực sự mang lại hiệu quả hay không.

_________________

---

## 2. Điều tôi hiểu rõ hơn sau lab này (100-150 từ)

> Chọn 1-2 concept từ bài học mà bạn thực sự hiểu rõ hơn sau khi làm lab.
> Ví dụ: chunking, hybrid retrieval, grounded prompt, evaluation loop.
> Giải thích bằng ngôn ngữ của bạn — không copy từ slide.

Sau khi làm lab này, tôi hiểu sâu sắc hơn về khái niệm **Evaluation Loop (Vòng lặp đánh giá)** và phương pháp **LLM-as-Judge**. Trước đây, tôi nghĩ việc đánh giá AI đơn giản là đọc thử vài câu trả lời xem có lọt tai không. Tuy nhiên, qua lab này, tôi hiểu rằng đánh giá RAG phải tách bạch rõ ràng giữa *Retrieval quality* và *Generation quality*. 

Bên cạnh đó, tôi thực sự hiểu rõ cách hoạt động của LLM-as-Judge. Thay vì phải chật vật viết code bằng Regex để chấm điểm text rất cứng nhắc, ta có thể viết một system prompt thật chặt chẽ, định nghĩa rõ thang điểm 1-5 và ép LLM trả về cấu trúc JSON (chứa điểm số và lý do trừ điểm). Cách này linh hoạt và giống với tư duy chấm bài của con người hơn rất nhiều.

_________________

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn (100-150 từ)

> Điều gì xảy ra không đúng kỳ vọng?
> Lỗi nào mất nhiều thời gian debug nhất?
> Giả thuyết ban đầu của bạn là gì và thực tế ra sao?


_________________

---

## 4. Phân tích một câu hỏi trong scorecard (150-200 từ)

> Chọn 1 câu hỏi trong test_questions.json mà nhóm bạn thấy thú vị.
> Phân tích:
> - Baseline trả lời đúng hay sai? Điểm như thế nào?
> - Lỗi nằm ở đâu: indexing / retrieval / generation?
> - Variant có cải thiện không? Tại sao có/không?

**Câu hỏi:** "q06 - Escalation trong sự cố P1 diễn ra như thế nào?"

**Phân tích:**
Ở phiên bản Baseline, mô hình xử lý tương đối tốt khi các chỉ số Faithfulness, Relevance và Context Recall đều đạt tối đa (5/5). Điều này chứng tỏ nó tìm đúng tài liệu trọng tâm và không bịa đặt thông tin. Tuy nhiên, điểm Completeness chỉ đạt 4/5, nghĩa là câu trả lời vẫn bị thiếu một chi tiết nhỏ so với đáp án chuẩn.
Nguyên nhân: dùng dense search và top k select (3) quá nhỏ khiến hệ thống đã cắt bỏ mất chunk chứa tiểu tiết phụ.
Variant đã cải thiện completeness lên 5/5. Nguyên nhân là chiến lược tìm kiếm đã đổi từ dense sang hybrid, cùng với việc nới rộng top_k_select lên 7 đã mở rộng "tầm nhìn" của mô hình. Nhờ có bộ ngữ cảnh dồi dào và đầy đủ hơn, khâu generation đã sinh ra được một câu trả lời bao phủ toàn bộ các góc cạnh của vấn đề.

_________________

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì? (50-100 từ)

> 1-2 cải tiến cụ thể bạn muốn thử.
> Không phải "làm tốt hơn chung chung" mà phải là:
> "Tôi sẽ thử X vì kết quả eval cho thấy Y."

Nếu có thêm thời gian, tôi sẽ thử nghiệm chiến lược Semantic Chunking thay vì chia nhỏ văn bản theo độ dài cố định. Kết quả eval cho thấy điểm Completeness đôi khi bị thấp do thông tin quan trọng bị cắt đôi giữa hai chunk, khiến LLM thiếu ngữ cảnh để trả lời trọn vẹn.
_________________

---


