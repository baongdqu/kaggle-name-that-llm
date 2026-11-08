import os
import sys

sys.path.append(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa")
from ask_claude import ask_claude

prompt = """
Chào Claude, bạn đang tham gia Hội đồng Cố vấn Cấp cao (Senior Consultant & Lead Architect) cho cuộc thi Kaggle "Name That LLM" (18 classes, metric Balanced Accuracy).

Người dùng vừa thốt lên với sự sốt ruột và mệt mỏi:
"claude ơi, bạn hãy phản biện lại coi thử, tôi ở các môc này lâu lắm r"

Bối cảnh và dữ liệu thực tế tại dự án hiện tại:
1. Leaderboard hiện tại:
   - V80-V89: quẩn quanh 0.663
   - V92 (Champion V92): đạt 0.664 (tăng đúng 0.001)
   - V93 (Champion V93): sụp đổ thảm hại xuống 0.528!
   - Trong khi đó, Local CV (5-Fold GroupKFold) của team đo được lên tới 72.73% (+9.72% so với baseline DL 63.01%)!
   - Top 1 thế giới (Le Viet) đang là 0.751, Top 10 thế giới là 0.724.

2. Chi tiết kỹ thuật vừa mổ xẻ được:
   - Tại sao Local CV đo được 72.73% nhưng V92 trên LB chỉ đạt 0.664?
     -> Trong V92, team chạy với cờ `--perpassage 1` trong `infer.py`. Dòng 905 viết:
        `pred = tot.argmax(1) if a.perpassage else hungarian(tot, Gte)`
        Vì `perpassage=1`, V92 hoàn toàn KHÔNG CHẠY Hungarian matching trên test set, mà chạy độc lập từng passage bằng `argmax(1)`!
        Khi ta đo lại trên tập train local mô phỏng test bằng `argmax(1)` độc lập, điểm chỉ là 49.38% (tương ứng với 0.664 trên LB). Còn khi chạy Hungarian với Oracle true prompt_id thì điểm nhảy vọt lên 72.73%!
   - Tại sao V93 bật `--perpassage 0` để ép chạy Hungarian thì LB lại sập xuống 0.528?
     -> Vì cờ `--perpassage 0` trong code cũ kích hoạt cùng lúc 3 quả bom:
        1. Nó gọi `center_groups(Ete, Gte)` làm biến dạng embedding Qwen/DeBERTa bằng cách trừ đi trung bình của các cluster vô giám sát với threshold cực thấp `0.02` (gộp hàng trăm prompt khác nhau thành các cụm khổng lồ >100 mẫu).
        2. Dòng 866 trong `infer.py` có một bug logic chết người (Fatal Sibling Truncation):
           `_cand_classes = [L2I[s] for s in _siblings[:len(_sib_indices)]]`
           Khi một cluster có 2 mẫu sister (ví dụ opus-4-8 và opus-5), code cắt `_siblings[:2]` ép cứng thành `['claude-opus-4-6', 'claude-opus-4-8']`, triệt tiêu hoàn toàn nhãn `claude-opus-5` trên toàn bộ test set! Tương tự làm biến mất `gpt-5.6-terra` và `gemini-3.8-flash`!

3. Yêu cầu phản biện từ bạn (Claude Sonnet):
Hãy đưa ra một bài phản biện chuyên sâu, sắc bén, thẳng thắn của một Cố vấn Kiến trúc Trưởng / Kaggle Grandmaster:
- Vạch trần sự ảo tưởng và điểm mù giữa Local CV (72.73% với Oracle Prompt) và Kaggle Test Set thực tế (Unlabeled, shuffled).
- Phân tích bản chất vì sao team bị "kẹt cứng" ở mốc 0.663-0.664 suốt thời gian dài.
- Giải phẫu kỹ thuật ghép cặp Bipartite Matching trên Test Set: Làm thế nào để gom cụm prompt (Unsupervised Prompt Clustering) chuẩn xác nhất mà không bị rò rỉ hay biến dạng embedding như V93?
- Chiến lược xử lý các cặp mô hình anh em (Opus 4.6/4.8/5, Gemini 3.7/3.8, GPT Sol/Terra): Tại sao Deep Learning thất bại và Classical ML cần can thiệp ở tầng nào (Pre-Hungarian hay Post-Hungarian)?
- Đưa ra bản thiết kế kiến trúc chuẩn xác nhất cho Champion V94 để phá vỡ ngay lập tức bức tường 0.664, đưa điểm LB nhảy vọt thẳng lên vùng 0.70 - 0.73+.

Hãy trình bày với năng lượng tích cực, thấu hiểu tâm lý sốt ruột của người dùng, nhưng tuyệt đối nghiêm cẩn, khoa học và thực chiến!
"""

print("Calling ask_claude...")
try:
    res = ask_claude(prompt, max_tokens=4096)
    content = res.get("content", "")
    with open(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\claude_critique_response.txt", "w", encoding="utf-8") as out_f:
        out_f.write(content)
    print("Success! Got", len(content), "characters.")
    print("Model:", res.get("model"))
except Exception as e:
    import traceback
    traceback.print_exc()
