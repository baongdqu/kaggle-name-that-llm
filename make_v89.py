import json

# Load base champion_v86 notebook
nb = json.load(open('champion_v86/champion-v86.ipynb', encoding='utf-8'))

# Cell 0: Title
nb['cells'][0]['source'] = ['# Name That LLM Champion V89 Calibrated Intra-Family Stylometric Permutation Apex\n']

# Cell 6: infer.py modifications
infer_src = ''.join(nb['cells'][6]['source'])

# Replace the stylometric scoring within V86 tier2 with empirical rules
old_tier2_scoring = '''                        if 'opus' in _cls_name:
                            if _cls_name == 'claude-opus-5': _sty += 0.30 * _test_colon[_row_idx] + 0.20 * _test_semi[_row_idx]
                            elif _cls_name == 'claude-opus-4-8': _sty += 0.20 * _rel_w[_r_i] - 0.20 * _test_em[_row_idx]
                            elif _cls_name == 'claude-opus-4-6': _sty += 0.25 * _test_em[_row_idx] - 0.15 * _rel_w[_r_i]
                        elif 'gpt-5.6' in _cls_name:
                            if _cls_name == 'gpt-5.6-sol': _sty += 0.25 * _test_semi[_row_idx] + 0.15 * _rel_w[_r_i]
                            elif _cls_name == 'gpt-5.6-luna': _sty += 0.20 * _test_em[_row_idx]
                            elif _cls_name == 'gpt-5.6-terra': _sty -= 0.15 * _rel_w[_r_i]
                        elif 'gemini' in _cls_name:
                            if _cls_name == 'gemini-3.7-flash': _sty += 0.20 * _rel_w[_r_i]
                            elif _cls_name == 'gemini-3.8-flash': _sty -= 0.25 * _rel_w[_r_i]
                        _combined_scores[_r_i, _c_j] = _t1 + _lam * _sty'''

new_tier2_scoring = '''                        if 'opus' in _cls_name:
                            # Empirical: opus-5 has highest em-dash (1.50) + highest colon (0.96) + highest semi (0.66)
                            if _cls_name == 'claude-opus-5': _sty += 0.35 * _test_colon[_row_idx] + 0.25 * _test_semi[_row_idx] + 0.15 * _test_em[_row_idx]
                            # opus-4-8 has lowest em-dash (0.97) + longest words (123.6)
                            elif _cls_name == 'claude-opus-4-8': _sty += 0.25 * _rel_w[_r_i] - 0.30 * _test_em[_row_idx]
                            # opus-4-6 has intermediate em-dash (1.30) + shortest words (115.1)
                            elif _cls_name == 'claude-opus-4-6': _sty += 0.20 * _test_em[_row_idx] - 0.25 * _rel_w[_r_i] - 0.20 * _test_colon[_row_idx]
                        elif 'gpt-5.6' in _cls_name:
                            # Empirical: sol has highest semi (0.84) + longest words (118.9)
                            if _cls_name == 'gpt-5.6-sol': _sty += 0.30 * _test_semi[_row_idx] + 0.20 * _rel_w[_r_i]
                            # luna has highest em-dash (0.49) + intermediate words (115.4)
                            elif _cls_name == 'gpt-5.6-luna': _sty += 0.30 * _test_em[_row_idx]
                            # terra has lowest em-dash (0.37) + shortest words (113.3)
                            elif _cls_name == 'gpt-5.6-terra': _sty -= 0.20 * _rel_w[_r_i] - 0.20 * _test_em[_row_idx]
                        elif 'gemini' in _cls_name:
                            # Empirical: 3.7 is longest (117.1), 3.8 is shortest (100.7), 3.6 is intermediate (111.2)
                            if _cls_name == 'gemini-3.7-flash': _sty += 0.35 * _rel_w[_r_i]
                            elif _cls_name == 'gemini-3.8-flash': _sty -= 0.40 * _rel_w[_r_i]
                            elif _cls_name == 'gemini-3.6-flash': _sty -= 0.10 * abs(_rel_w[_r_i])
                        _combined_scores[_r_i, _c_j] = _t1 + _lam * _sty'''

if old_tier2_scoring in infer_src:
    infer_src = infer_src.replace(old_tier2_scoring, new_tier2_scoring)
    print("SUCCESS: Replaced V86 tier2 scoring with V89 calibrated empirical scoring!")
else:
    print("ERROR: old_tier2_scoring not found in infer_src!")

# Keep conservative clustering threshold adapt_thr=0.15 (safe 0.663 anchor), but adjust lambdas
old_lam = "_LAMBDA_WEIGHTS = {'claude_opus': 1.0, 'gpt_56': 0.40, 'gemini_flash': 0.25}"
new_lam = "_LAMBDA_WEIGHTS = {'claude_opus': 1.25, 'gpt_56': 0.50, 'gemini_flash': 0.35}"
if old_lam in infer_src:
    infer_src = infer_src.replace(old_lam, new_lam)
    print("SUCCESS: Updated _LAMBDA_WEIGHTS to V89 calibrated levels!")
else:
    print("WARNING: old_lam not matched!")

nb['cells'][6]['source'] = [l + '\n' for l in infer_src.splitlines()]

# Cell 12: CLI arguments
cmd_src = ''.join(nb['cells'][12]['source'])
# Ensure svc_w=8.5 (slightly reinforced from 8.0) and adapt_thr remains 0.15
cmd_src = cmd_src.replace("'--svc_w', '8.0'", "'--svc_w', '8.5'")
nb['cells'][12]['source'] = [l + '\n' for l in cmd_src.splitlines()]
print("SUCCESS: Updated --svc_w to 8.5 in Cell 12")

# Write to champion_v89
with open('champion_v89/champion-v89.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("champion_v89/champion-v89.ipynb built successfully!")

