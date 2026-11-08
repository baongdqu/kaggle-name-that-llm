import sys
import json
import urllib.request

sys.path.append(r"C:\Users\s3cr3t\.gemini\antigravity\scratch\kaggle_claude_qa")
from claude_session import load_env, SYSTEM_PROMPT

v68_prompt = """[BÁO CÁO KẾT QUẢ ĐÊM QUA & THẨM ĐỊNH KẾ HOẠCH V68]
Chào Claude! Sáng nay chúng tôi đã nhận được kết quả chấm điểm của 3 phiên bản đêm qua:
- V65 (Llama w=2.4 thay vì 1.8): Đạt 0.657 (-0.005 so với 0.662 ở V63). Chứng minh trọng số Llama 3.1 8B ở w=1.8 của V63 đã là điểm ngọt tối ưu.
- V66 & V67 (Hạ ngưỡng gom cụm adapt_thr từ 0.15 xuống 0.08 với bigrams): Đều đạt 0.647 (-0.015). Phân tích cho thấy: Trên test ẩn, ngưỡng 0.08 gây ra 'False-Merges' (gộp nhầm các prompt khác nhau cùng chủ đề), làm hỏng phép trừ trung bình cụm (center_groups) trong Causal Decoders. Do đó, gom cụm PHẢI giữ nguyên ngưỡng bảo thủ thr=0.15 của V63.
- Thứ hạng chính thức vẫn được bảo vệ 100% tại mốc sàn kỷ lục 0.662 🥉👑 (Rank 312 / 4,150 đội).
- Hôm nay (03/11 UTC), quota nộp bài đã RESET về 5/5 lượt tươi mới!

ĐẶC BIỆT: Đúng như bạn đã chỉ ra đêm qua về '14 lớp sinh đôi', chúng tôi vừa lập Confusion Matrix trên toàn bộ tập train:
- Lỗi nhầm lẫn nội bộ dòng họ (Sibling Confusion) chiếm đúng 49.3% (gần một nửa) tổng số lỗi!
- Top 1 gây lỗi là Gemini Flash siblings: gemini-3.7-flash <-> gemini-3.8-flash (93 lỗi), 3.6-flash <-> 3.7-flash (69 lỗi).
- Top 2 là GPT siblings: gpt-5.6-luna <-> 5.6-sol (54 lỗi), 5.6-luna <-> 6-astra (53 lỗi).
- Top 3 là Claude siblings: claude-fable-5 <-> opus-5 (41 lỗi), opus-4-8 <-> opus-5 (33 lỗi).

Chúng tôi đã thiết kế đột phá cho Champion V68:
TRI-ATTRIBUTE PROMPT-RELATIVE CALIBRATION:
Ngoài z-score độ dài từ (z_len) đã có ở V63, bổ sung thêm 2 chiều phong cách tương đối:
1. z_avg_wlen (Độ dài từ trung bình tương đối - bắt trúng từ vựng đơn giản hay cao cấp: Claude Opus-5 có z=-1.102 cực thấp trong khi Sonnet-5 có z=+0.598!).
2. z_ttr (Type-Token Ratio tương đối - độ phong phú từ vựng: Gemini 3.8-flash có z=+0.444 trong khi 3.1-pro có z=-0.287!).
-> Thử nghiệm Digital Twin Lab trên 50 kịch bản test lát cắt 40%: Đem lại mức tăng +0.0060 (+0.60%) với tỷ lệ thắng đối đầu 86.0%!

Claude ơi, bạn đánh giá thế nào về thiết kế V68 này? Bạn có khuyến nghị tinh chỉnh nào về trọng số hoặc cơ chế bảo vệ trước khi chúng tôi push và submit V68 không?
"""

env = load_env()
api_key = env.get("MODEL_PROXY_API_KEY")
endpoint = "https://mp-staging.kaggle.net/models/openapi/chat/completions"

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": v68_prompt}
]

payload = {
    "model": "anthropic/claude-sonnet-5@default",
    "messages": messages,
    "max_tokens": 4096,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 1024
    }
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

print("Đang gửi kế hoạch V68 tới Claude Sonnet...", flush=True)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res["choices"][0]["message"]
        reply = msg.get("content")
        print("\n" + "="*80)
        print("PHẢN HỒI THẨM ĐỊNH TỪ CLAUDE SONNET CHO V68:")
        print("="*80 + "\n")
        print(reply)
        with open('scratch/claude_v68_feedback.txt', 'w', encoding='utf-8') as fo:
            fo.write(reply)
except Exception as e:
    import traceback
    traceback.print_exc()

