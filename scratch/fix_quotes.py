import json

with open('champion_v70/champion-v70.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][6]['source'])

bad_quote_line = "_quote = (_st.count('\"') + _st.count(\\\"'\") + _st.count('“') + _st.count('”')) / _nw"
good_quote_line = "_quote = (_st.count(chr(34)) + _st.count(chr(39)) + _st.count('“') + _st.count('”')) / _nw"

# Replace directly
for line in src.split('\n'):
    if '_quote =' in line:
        src = src.replace(line, "                " + good_quote_line)
        print("Replaced quote line with clean chr() version!")
        break

nb['cells'][6]['source'] = [src]

with open('champion_v70/champion-v70.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Saved clean notebook!")

