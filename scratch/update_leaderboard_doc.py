with open('abc LEADERBOARD.md', 'r', encoding='utf-8') as f:
    text = f.read()

old_sec5 = """## 5. KẾ HOẠCH HÀNH ĐỘNG CHO 2 LƯỢT NỘP CÒN LẠI HÔM NAY (01/11/2026 UTC)

Hiện tại đội ta đã dùng 3/5 lượt nộp. **Còn đúng 2 lượt nộp nguyên vẹn**:
1. **Champion V61 (The 0.664+ Silver Breaker):**
   - Kế thừa toàn bộ lõi V59 (Prompt-Aware Length Bias $\\beta_m$ + Soft Collision Dispersion 0.65).
   - Tích hợp công thức vàng của Judith007: **Sinkhorn = 0.60**, **Evidence Clip = 1.2**.
   - Tăng trọng số của **Qwen 2.5 14B (`q25_14`) từ 1.0 lên 2.0**.
   - *Mục tiêu:* Đánh thẳng lên **`0.664+`**, vượt qua 101 đội ở dải 0.663 để nhảy vào **Top 150 (Vùng Huy Chương Bạc 🥈 vững chắc)**!
2. **Champion V62 (The GLM-4 Frontier Titan):**
   - Bổ sung backbone **GLM-4 9B** vào ensemble để phá vỡ điểm mù Claude vs GLM.
   - *Mục tiêu:* Mở đường tiếp cận mốc **`0.670+`** (Vùng Huy chương Vàng)."""

new_sec5 = """## 5. TRẠNG THÁI & KẾ HOẠCH HÀNH ĐỘNG HÔM NAY (01/11/2026 UTC)

Hiện tại đội ta đã dùng **4/5 lượt nộp** (V58, V59, V60, V61). **Còn đúng 1 lượt nộp duy nhất**:
1. **Champion V61 (Judith Sovereign Apex - ĐÃ NỘP & ĐANG CHẤM `PENDING` - Ref: 57868985):**
   - Kế thừa toàn bộ lõi vững chãi V59 (Prompt-Aware Length Bias $\\beta_m$ + Soft Collision Dispersion 0.65).
   - Tích hợp trọn vẹn công thức vàng bứt phá của Judith007: **Sinkhorn = 0.60**, **Evidence Clip = 1.2**.
   - Tăng trọng số của **Qwen 2.5 14B (`q25_14`) từ 1.0 lên 2.0**.
   - Tuân thủ nghiêm ngặt kỷ luật chống Hungarian gượng ép (`--perpassage 1`).
   - *Mục tiêu:* Đánh thẳng lên **`0.664+`**, vượt qua 101 đội ở dải 0.663 để nhảy vào **Top 150 (Vùng Huy Chương Bạc 🥈 vững chắc)**!
2. **Champion V62 (Dự phòng / Bứt phá tiếp theo theo chỉ đạo):**
   - Tạm thời giữ nguyên quota cuối cùng chờ kết quả điểm số thực tế của V61 từ Kaggle.
   - Khi sẵn sàng: Tích hợp GLM-4 9B hoặc triển khai module Zero-Shot Perplexity Forensics / Meta-Stacking để phá vỡ điểm mù các họ mô hình."""

text = text.replace(old_sec5, new_sec5)

with open('abc LEADERBOARD.md', 'w', encoding='utf-8') as f:
    f.write(text)

print('abc LEADERBOARD.md updated!')

