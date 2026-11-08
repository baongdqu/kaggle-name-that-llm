with open("abc CHAT_HISTORY.md", "r", encoding="utf-8") as f:
    text = f.read()

# Replace tab followed by imes with \\times
fixed_text = text.replace("0.15\times", "0.15$\\times$").replace("0.20\times", "0.20$\\times$")
if "\times" in text:
    fixed_text = fixed_text.replace("\times", "$\\times$")

with open("abc CHAT_HISTORY.md", "w", encoding="utf-8") as f:
    f.write(fixed_text)

print("Formatting cleaned up in abc CHAT_HISTORY.md")
