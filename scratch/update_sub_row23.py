with open("abc SUBMISSIONS.md", "r", encoding="utf-8") as f:
    text = f.read()

old_row23 = "| **23** | 2026-10-25 10:42 | `LOCAL-READY` | **Champion V23 Curated Sovereign**: Route-Filtered 86.5k Curated Pool (`ankiteshtiwari`) + Lossless Speedup Trio A+B+C + Terminal Sinkhorn | *N/A* | *N/A* | *Chờ nộp* | `LOCAL_READY` | Đã code xong và kiểm thử AST + mô phỏng 4,000 dòng test 100% PASS. Nạp 81,327 mẫu route-filtered sạch chuẩn + Bộ ba tối ưu tốc độ Lossless (A+B+C) rút ngắn ~10-15 phút. Chưa push/submit theo chỉ thị của User. |"

new_row23 = "| **23** | 2026-10-25 11:22 | `LOCAL-READY` | **Champion V23 Perfected**: Curated Pool 86.5k + Lossless Trio + Twin `min_size=4` + Soft Pre-Sinkhorn TypoGuard + Terminal Sinkhorn | *N/A* | *N/A* | *Chờ nộp* | `LOCAL_READY` | Đã build và kiểm thử AST + mô phỏng 4,000 dòng test 100% PASS. Nạp 81,327 mẫu certified + Sửa lỗi đại số Twin Centering (`min_size=4` chặn đứng mất -0.067 BA trên cặp $k=2$) + Soft Pre-Sinkhorn TypoGuard (0.15 Claude/GLM, 0.15 Gemini, 0.20 GPT) + Bộ ba tăng tốc Lossless (A+B+C) + Terminal Sinkhorn Purity ($N/18$). Sẵn sàng trên máy, chưa push/submit theo chỉ đạo của User. |"

if old_row23 in text:
    text = text.replace(old_row23, new_row23)
    with open("abc SUBMISSIONS.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("SUCCESS: abc SUBMISSIONS.md Row 23 updated!")
else:
    print("WARNING: old_row23 not found in abc SUBMISSIONS.md!")
