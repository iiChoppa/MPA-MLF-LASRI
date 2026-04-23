import pdfplumber
import re

pdf_path = "MLF_MPA_2026__Final_Project_ISEP.pdf"

with pdfplumber.open(pdf_path) as pdf:
    full_text = ""
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

# Search for keywords (case-insensitive)
keywords = ["github", "repository", "git", "dépôt", "code repository", "source code"]
pattern = "|".join(keywords)

matches = list(re.finditer(pattern, full_text, re.IGNORECASE))

if matches:
    print(f"Found {len(matches)} mentions of GitHub/repository/git/dépôt/code repository/source code:\n")
    
    # Extract context around matches (150 characters before and after)
    seen_contexts = set()
    for i, match in enumerate(matches):
        start = max(0, match.start() - 150)
        end = min(len(full_text), match.end() + 150)
        context = full_text[start:end].strip().replace("\n", " ")
        
        if context not in seen_contexts:
            print(f"\n--- Match {len(seen_contexts)+1} ---")
            print(context)
            print()
            seen_contexts.add(context)
else:
    print("No mentions of GitHub/repository/git/dépôt/code repository/source code found in the PDF.")
