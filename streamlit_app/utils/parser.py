from bs4 import BeautifulSoup
import re

SEMANTIC_TAGS = ["main", "article", "section"]

def clean_text(t: str) -> str:
    if not t:
        return ""
    # Lowercase, collapse whitespace
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_title_and_text(html: str):
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "lxml")
    # Remove scripts/styles/nav/aside/footer/head
    for tag in soup(["script", "style", "nav", "aside", "footer", "noscript"]):
        tag.decompose()
    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    # Prefer semantic containers; fallback to paragraphs
    for name in SEMANTIC_TAGS:
        node = soup.find(name)
        if node:
            text = " ".join(p.get_text(" ", strip=True) for p in node.find_all(["p", "li"]))
            if len(text.split()) > 50:
                return title, text
    # Fallback: all paragraphs
    text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
    return title, text
