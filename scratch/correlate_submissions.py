import json
import re
from datetime import datetime

transcript_path = r"C:\Users\s3cr3t\.gemini\antigravity-ide\brain\f5ca0093-20f3-4308-a03b-7ccba8e6503a\.system_generated\logs\transcript.jsonl"

subs = {
    "57285498": {"v": "V1", "submit_time": "2026-10-16 22:50:55", "last_pending": None, "first_complete": None},
    "57285636": {"v": "V2", "submit_time": "2026-10-16 22:57:57", "last_pending": None, "first_complete": None},
    "57290359": {"v": "V3", "submit_time": "2026-10-17 01:28:21", "last_pending": None, "first_complete": None},
    "57294735": {"v": "V4 (Twin)", "submit_time": "2026-10-17 04:15:54", "last_pending": None, "first_complete": None},
    "57294743": {"v": "V4", "submit_time": "2026-10-17 04:16:05", "last_pending": None, "first_complete": None},
    "57294881": {"v": "V5", "submit_time": "2026-10-17 04:21:04", "last_pending": None, "first_complete": None},
    "57296505": {"v": "V6", "submit_time": "2026-10-17 05:18:17", "last_pending": None, "first_complete": None},
    "57330867": {"v": "V7", "submit_time": "2026-10-18 00:09:37", "last_pending": None, "first_complete": None},
    "57330913": {"v": "V8", "submit_time": "2026-10-18 00:10:20", "last_pending": None, "first_complete": None},
    "57336421": {"v": "V9", "submit_time": "2026-10-18 03:20:08", "last_pending": None, "first_complete": None},
    "57336428": {"v": "V10", "submit_time": "2026-10-18 03:20:19", "last_pending": None, "first_complete": None},
    "57359637": {"v": "V11", "submit_time": "2026-10-18 15:27:32", "last_pending": None, "first_complete": None},
    "57373323": {"v": "V12", "submit_time": "2026-10-19 00:03:06", "last_pending": None, "first_complete": None},
    "57383708": {"v": "V13", "submit_time": "2026-10-19 05:23:30", "last_pending": None, "first_complete": None},
    "57388892": {"v": "V14", "submit_time": "2026-10-19 08:04:48", "last_pending": None, "first_complete": None},
    "57394349": {"v": "V15", "submit_time": "2026-10-19 11:21:33", "last_pending": None, "first_complete": None},
    "57399922": {"v": "V16", "submit_time": "2026-10-19 14:47:09", "last_pending": None, "first_complete": None},
    "57580018": {"v": "V17", "submit_time": "2026-10-24 02:46:41", "last_pending": None, "first_complete": None},
}

with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            entry = json.loads(line)
            ts = entry.get("created_at", "")
            content = str(entry.get("content", ""))
            for ref, data in subs.items():
                if ref in content:
                    # check if this line has ref with PENDING
                    lines = content.splitlines()
                    for l in lines:
                        if ref in l:
                            if "SubmissionStatus.PENDING" in l:
                                data["last_pending"] = ts
                            elif "SubmissionStatus.COMPLETE" in l:
                                if data["first_complete"] is None:
                                    data["first_complete"] = ts
        except:
            pass

print(f"{'Ref':<10} {'Ver':<12} {'Submit (UTC)':<20} {'Last Pending':<24} {'First Complete':<24} {'Scoring Time':<18}")
for ref, data in subs.items():
    t_sub = datetime.strptime(data["submit_time"], "%Y-%m-%d %H:%M:%S")
    lp = data["last_pending"]
    fc = data["first_complete"]
    
    scoring_time_str = "N/A"
    if fc:
        fc_clean = fc.replace("Z", "").split(".")[0]
        t_comp = datetime.strptime(fc_clean, "%Y-%m-%dT%H:%M:%S")
        diff_m = round((t_comp - t_sub).total_seconds() / 60.0, 1)
        scoring_time_str = f"~{diff_m} min"
        if lp:
            lp_clean = lp.replace("Z", "").split(".")[0]
            t_lp = datetime.strptime(lp_clean, "%Y-%m-%dT%H:%M:%S")
            diff_lp = round((t_lp - t_sub).total_seconds() / 60.0, 1)
            scoring_time_str += f" (pending at {diff_lp}m)"
    elif lp:
        lp_clean = lp.replace("Z", "").split(".")[0]
        t_lp = datetime.strptime(lp_clean, "%Y-%m-%dT%H:%M:%S")
        diff_lp = round((t_lp - t_sub).total_seconds() / 60.0, 1)
        scoring_time_str = f"STILL PENDING ({diff_lp}m)"
        
    print(f"{ref:<10} {data['v']:<12} {data['submit_time']:<20} {str(lp):<24} {str(fc):<24} {scoring_time_str:<18}")
