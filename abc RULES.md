# Name That LLM - Competition Rules & Compliance (RULE.md)

## 1. Thông tin pháp lý & Nhà tài trợ (Sponsorship & Legal)
- **Tên cuộc thi:** Name That LLM - Identify the Model Behind the Text
- **Nhà tài trợ chính (Sponsor):** **Google LLC** (1600 Amphitheatre Parkway, Mountain View, CA 94043)
- **Nền tảng tổ chức (Host):** Kaggle Inc.
- **Giấy phép mã nguồn người thắng giải (Winner License):** **CC-BY 4.0** (Open Source được OSI phê duyệt, không giới hạn mục đích thương mại).
- **Giấy phép dữ liệu (Data License):** **CC BY-SA 4.0** (Được phép sử dụng cho nghiên cứu, học thuật, thương mại và phi thương mại).
- **Luật điều chỉnh:** Luật bang California, Hoa Kỳ (Tòa án Quận Santa Clara).

---

## 2. Quy định Đội thi & Giới hạn nộp bài (Teams & Submissions)

| Quy định | Chi tiết giới hạn | Lưu ý quan trọng |
| :--- | :--- | :--- |
| **Quy mô đội tối đa (Max Team Size)** | **5 người / đội** | Mỗi thành viên phải là 1 tài khoản cá nhân duy nhất. |
| **Ghép đội (Team Mergers)** | Cho phép trước hạn chót | Tổng số lượt nộp của các đội gộp lại không được vượt quá số lượt tối đa tính đến thời điểm ghép. |
| **Giới hạn nộp bài mỗi ngày** | **5 bài nộp / ngày** (5 Submissions/day) | Reset vào 00:00 UTC hàng ngày. |
| **Số bài chọn chấm chung cuộc** | **Tối đa 2 bài** (2 Final Submissions) | Nếu không tự chọn, hệ thống sẽ tự động chọn bài có điểm Public LB cao nhất. |
| **Một tài khoản duy nhất** | Nghiêm cấm dùng nhiều tài khoản (Cheating/Multi-accounting) | Vi phạm sẽ bị truất quyền thi đấu vĩnh viễn (Disqualification). |

---

## 3. Quy định về Mã nguồn & Chia sẻ riêng tư (Code & Private Sharing)

> [!CAUTION]
> **TUYỆT ĐỐI KHÔNG CHIA SẺ RIÊNG TƯ (NO PRIVATE SHARING):**
> Nghiêm cấm mọi hình thức chia sẻ code, dữ liệu, trọng số mô hình hoặc gợi ý kết quả bên ngoài phạm vi các thành viên trong cùng một đội (trừ khi hai đội chính thức sáp nhập).

- **Chia sẻ công khai (Public Code Sharing):**
  - Được phép và khuyến khích chia sẻ trên diễn đàn hoặc public Kaggle Notebooks của cuộc thi.
  - Mã nguồn chia sẻ công khai mặc định tuân theo giấy phép mã nguồn mở (OSI-approved Open Source license).
- **Phần mềm AutoML:**
  - Cho phép sử dụng các công cụ AutoML (Google AutoML, H2O Driverless AI,...) miễn là thí sinh có giấy phép hợp lệ và đáp ứng nghĩa vụ bàn giao giải pháp.

---

## 4. Quy định về Dữ liệu ngoại vi & Pretrained Models (External Data & Models)

Các điều khoản về dữ liệu ngoại vi và chi phí được quy định cực kỳ rõ ràng:

1. **Tính công khai & Bình đẳng (Public Availability):**
   - Mọi dữ liệu ngoại vi (External Data) hay pretrained model sử dụng **bắt buộc phải công khai và mọi người tham gia đều có thể tiếp cận bình đẳng, miễn phí**.
   - Nếu bạn tự tạo dataset (ví dụ gọi API 18 model để sinh thêm passage), bạn **phải đăng công khai dataset đó trên Kaggle Datasets** trước hạn chót.
2. **Tiêu chuẩn Hợp lý về Chi phí (Reasonableness Standard):**
   - Nghiêm cấm các công cụ hoặc dataset độc quyền có chi phí vượt quá mức hợp lý (ví dụ mua dataset vượt quá giá trị tiền thưởng $5,000).
   - Cho phép các gói thuê bao nhỏ phổ biến (như Gemini Advanced hoặc API quota thông thường) miễn là tuân thủ tiêu chuẩn "Reasonableness".
3. **Cấm gán nhãn thủ công (No Hand Labeling):**
   - **Nghiêm cấm tuyệt đối** việc gán nhãn thủ công bằng con người (human hand labeling) trên tập validation hoặc test set.

---

## 5. Quy định giải thưởng & Nghĩa vụ người chiến thắng (Winner Obligations)

- **Cơ cấu giải thưởng:** $50,000 USD (Top 1 đến Top 10 nhận đồng đều **$5,000 USD/giải**).
- **Điều kiện nhận giải:**
  1. Bàn giao đầy đủ mã nguồn huấn luyện (training code), mã nguồn dự đoán (inference code), cấu hình môi trường và tài liệu hướng dẫn tái lập kết quả (Reproducibility guidelines).
  2. Ký thỏa thuận cấp phép mã nguồn mở theo chuẩn CC-BY 4.0 / OSI-approved.
  3. Hoàn tất các biểu mẫu thuế Hoa Kỳ (IRS W-9 cho công dân Mỹ hoặc IRS W-8BEN cho thí sinh quốc tế) trong vòng 2 tuần kể từ khi nhận thông báo.
  4. Tham gia buổi phỏng vấn/thuyết trình kỹ thuật (panel/call) với Google LLC và Kaggle nếu được yêu cầu.

---

## 6. Danh sách kiểm tra tuân thủ cho dự án (Compliance Checklist)

- [x] Đã chấp nhận điều khoản cuộc thi (Rules Accepted).
- [ ] Chỉ sử dụng 1 tài khoản Kaggle duy nhất.
- [ ] Không chia sẻ code/trọng số riêng tư cho bất kỳ ai ngoài team.
- [ ] Đảm bảo mọi pretrained backbone (Qwen, Gemma, v.v.) và dataset bổ sung đều được công khai trên Kaggle.
- [ ] Notebook submit chạy $\le 9$ giờ, đã tắt Internet và xuất đúng file `submission.parquet`.
- [ ] Không thực hiện hand-labeling trên dữ liệu test.
