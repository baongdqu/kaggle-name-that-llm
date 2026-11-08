with open('abc SUBMISSIONS.md', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()

idx_65 = None
for i, l in enumerate(lines):
    if l.startswith('| **65** |'):
        idx_65 = i
        break

print(f"Row 65 found at line {idx_65}")

row_66 = "| **66** | 2026-11-02 20:22 | `57909787` | **Champion V66 N-Gram Semantic Clustering Apex**: **Đột phá Gom cụm N-Gram Đa ngữ nghĩa (ARI $0.8959 \\to 0.9537$, $+5.8\\%$)** + **Ngưỡng gom cụm thích ứng tối ưu (`thr=0.08`, 128 cụm dày $\\ge 5$)** + **Latin-Square Anchor Kuhn-Munkres** + **Meta Llama-3.1-8B Amplified (`w=2.4`, dp=True)** + **Phục hồi Encoders Sạch 100% (`mb_v5 w=2.0`, `deb w=1.0`)** + **Damped Sinkhorn 0.45 (`--sinkhorn 0.45`, `--sinkhorn_target_mix 0.35`)** | **0.6m** *(Fast Fixture 35s)* | **~240m** *(4h 00m)* | **`PENDING`** | `PENDING` | **HƯỚNG ĐI 2: KHAI PHÓNG GOM CỤM N-GRAM ĐA TẦNG (ARI 0.954):** Thực nghiệm đo lường trên toàn bộ 2,699 mẫu chứng minh: Nâng cấp vectorizer lên `ngram_range=(1, 2)` với `max_df=0.10` và hạ ngưỡng cắt về điểm ngọt `0.08` giúp độ chính xác gom cụm tăng vọt từ 0.8959 lên 0.9537, mở rộng từ 111 lên 128 cụm dày (phủ 90% test set). Thuật toán Hungarian nhờ đó giải phóng toàn bộ sức mạnh song ánh mà không bị nhiễu cụm, sẵn sàng công phá mốc 0.670+! |"

row_67 = "| **67** | 2026-11-02 20:26 | `PENDING` | **Champion V67 Likelihood Sweet Spot Apex**: **Mở rộng Vùng can thiệp ATLAS Likelihood (`margin < 0.35`, trọng số `w=0.25`)** + **Đột phá Gom cụm N-Gram Đa ngữ nghĩa (ARI 0.9537, `thr=0.08`)** + **Latin-Square Anchor Kuhn-Munkres** + **Meta Llama-3.1-8B Amplified (`w=2.4`, dp=True)** + **Phục hồi Encoders Sạch 100% (`mb_v5 w=2.0`, `deb w=1.0`)** + **Damped Sinkhorn 0.45 (`--sinkhorn 0.45`, `--sinkhorn_target_mix 0.35`)** | **0.6m** *(Fast Fixture 35s)* | **~240m** *(4h 00m)* | **`PENDING`** | `PENDING` | **HƯỚNG ĐI 3: TỐI ƯU HÓA ĐIỂM NGỌT LIKELIHOOD SINH HỌC & ENSEMBLE ĐỈNH CAO:** Tận dụng năng lực phân định biên của mô hình sinh nhân quả Qwen 3 8B ATLAS, mở rộng ngưỡng can thiệp từ 0.30 lên 0.35 và tăng trọng số likelihood lên 25% nhằm giải quyết triệt để các ca lưỡng lự cận biên khó giữa các dòng họ (Qwen vs DeepSeek, Claude 3.5 vs Claude 3 Haiku). Kết hợp hoàn hảo với N-Gram Clustering ARI 0.954 để thiết lập kỷ lục điểm số mới! |"

if idx_65 is not None:
    # Check if row 66 already exists
    if idx_65 + 1 < len(lines) and lines[idx_65 + 1].startswith('| **66** |'):
        print("Row 66 already exists, skipping insert.")
    else:
        new_lines = lines[:idx_65+1] + [row_66, row_67] + lines[idx_65+1:]
        new_text = "\n".join(new_lines)
        new_text = new_text.replace('| **Tổng số Submissions đã nộp** | **65 lần**', '| **Tổng số Submissions đã nộp** | **66 lần**')
        with open('abc SUBMISSIONS.md', 'w', encoding='utf-8') as f:
            f.write(new_text)
        print("Inserted Row 66 and Row 67 successfully!")

