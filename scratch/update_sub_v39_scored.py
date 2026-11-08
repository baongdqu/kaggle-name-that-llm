with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update top score and status
text = text.replace("Public Score: 0.649", "Public Score: 0.653")
text = text.replace("`0.649` 🥈👑 (Thiết lập tại Champion V37 Llama Apex)", "`0.653` 🥈👑 (Thiết lập tại Champion V39 Llama Amplifier)")
text = text.replace("V37 (`0.649`)", "V39 (`0.653`)")

# Update table row 39
old_row_39 = "| **39** | 2026-10-28 19:43 | `57736884` | **Champion V39 Llama Amplifier**: Meta Llama-3.1-8B ($w=2.4$ Amplified) + Re-balanced Qwen 14B/8B ($w=2.8/1.8$) + Dual Encoders (MB 2.4 / Deb 1.6) + 4-View Stylometrics ($w=0.22$) | **2.2m** *(Fast Fixture 30s)* | *Đang chấm điểm...* | **PENDING** | `PENDING` | **CHÍNH THỨC NỘP BÀI - KHUẾCH ĐẠI LLAMA-3.1-8B LÊN TRỤ CỘT CHÍNH ($w=2.4$, +140%)!** Nâng tầm Llama-3.1-8B từ trọng số thử nghiệm 1.0 (V37 đạt 0.649) lên 2.4, phát huy tối đa tín hiệu trực giao phi-Qwen và tập từ vựng 128k trên không gian 18 lớp phẳng. Runtime dự kiến ~6.5 - 7.0 giờ, hoàn toàn an toàn trước giới hạn 9h. Mục tiêu: Bứt phá lên `0.655 - 0.662+`! |"

new_row_39 = "| **39** | 2026-10-28 19:43 | `57736884` | **Champion V39 Llama Amplifier**: Meta Llama-3.1-8B ($w=2.4$ Amplified) + Re-balanced Qwen 14B/8B ($w=2.8/1.8$) + Dual Encoders (MB 2.4 / Deb 1.6) + 4-View Stylometrics ($w=0.22$) | **2.2m** *(Fast Fixture 30s)* | **~310m** *(5h 10m)* | **`0.653`** 🥈👑 | `COMPLETE` | **ĐẠI ĐỘT PHÁ LỊCH SỬ (+0.004 SO VỚI V37, +0.007 SO VỚI V32) - CHÍNH THỨC BƯỚC VÀO VÙNG 0.653!** Đạt Rank 144 toàn cầu. Quyết định táo bạo khuếch đại Meta Llama-3.1-8B lên $w=2.4$ đã giải phóng toàn bộ năng lực trực giao của tập từ vựng 128k và biểu diễn RoPE theta 500k, tạo ra bước nhảy vọt ngoạn mục thứ 2 liên tiếp! |"

text = text.replace(old_row_39, new_row_39)

with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated abc SUBMISSIONS.md with V39 score 0.653 successfully")

