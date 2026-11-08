import json

with open("champion_v27/champion-v27.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

src6 = "".join(nb["cells"][6]["source"])
# In cell 6: Path('/kaggle/working/comp_stack.py').write_text(...)
# Let's extract the string passed to write_text
import re
match = re.search(r"Path\('/kaggle/working/comp_stack\.py'\)\.write_text\((.*)\)", src6, re.DOTALL)
if match:
    code_repr = match.group(1)
    # evaluate or unescape
    import ast
    try:
        code_str = ast.literal_eval(code_repr)
    except Exception as e:
        code_str = code_repr
    with open("scratch/comp_stack_dump.py", "w", encoding="utf-8") as f_out:
        f_out.write(code_str)
    print("Successfully wrote scratch/comp_stack_dump.py, length:", len(code_str))
else:
    print("Could not match write_text in cell 6!")
    # Just write the raw cell
    with open("scratch/comp_stack_dump.py", "w", encoding="utf-8") as f_out:
        f_out.write(src6)
    print("Wrote raw cell 6 to scratch/comp_stack_dump.py")
