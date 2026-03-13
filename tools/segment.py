import fitz
import re
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# ── LLM Segmentation ──────────────────────────────
SEGMENT_PROMPT = """
You are an NDA document processor specialising in Indian employment contracts.

Extract all clauses from the NDA and return structured JSON.

For each clause:
- clause_number  : sequential integer
- clause_title   : heading or title of the clause
- clause_text    : FULL original text, word for word — never summarize
- clause_type    : one of:
                   confidentiality | non-compete | non-solicitation |
                   penalty | jurisdiction | termination | ip-ownership |
                   data-protection | indemnity | definitions | general | other

Rules:
- Never paraphrase or summarize clause_text
- Include all sub-clauses inside clause_text
- If type unclear use "general"
- Return ONLY valid JSON — no markdown, no explanation

JSON structure:
{{
  "nda_title": "...",
  "total_clauses": <number>,
  "clauses": [
    {{
      "clause_number": 1,
      "clause_title": "...",
      "clause_text": "...",
      "clause_type": "..."
    }}
  ]
}}

NDA DOCUMENT:
{nda_text}
"""


# ── Regex Segmentation (fallback) ─────────────────
CLAUSE_TYPES = {
    "confidential"  : "confidentiality",
    "non-compete"   : "non-compete",
    "non compete"   : "non-compete",
    "non-solicit"   : "non-solicitation",
    "penalty"       : "penalty",
    "liquidated"    : "penalty",
    "jurisdiction"  : "jurisdiction",
    "governing law" : "jurisdiction",
    "terminat"      : "termination",
    "term "         : "duration",       
    "duration"      : "duration",
    "intellectual"  : "ip-ownership",
    "proprietary"   : "ip-ownership",
    "indemnif"      : "indemnity",
    "definition"    : "definitions",
    "data protect"  : "data-protection",
    "privacy"       : "data-protection",
}

# ── Patterns ──────────────────────────────────────

header_pattern = re.compile(
    r'(?m)^\s*'
    r'(?:'
        r'(?:Article|Section|Clause|Schedule)\s+\d+[\.\d]*'
        r'|\(\d+\)'
        r'|\d+[\.\d]*\)'
        r'|\d+[\.\d]*\.'
    r')'
    r'\s*'
    r'([A-Z][A-Za-z\s\-]{1,60})'
    r'\s*(?:—|-|:|\.|\n)',
    re.MULTILINE
)

list_item_pattern = re.compile(
    r'\n\s*\((?:\d+|[a-zA-Z]|[ivxIVX]+)\)\s+(.*?)'
    r'(?=\n\s*\((?:\d+|[a-zA-Z]|[ivxIVX]+)\)|\Z)',
    re.DOTALL
)

definition_pattern = re.compile(
    r'["\u201c\u2018](?P<term>.*?)["\u201d\u2019]'
    r'\s+(?:shall mean|shall include|shall have the meaning|means|includes)\s+'
    r'(?P<definition>.*?)'
    r'(?=\n[A-Z"\u201c]|\Z)',
    re.DOTALL
)

noise_cleanup = [
    (r'(?i)Page\s+\d+\s+of\s+\d+',              ''),
    (r'(?i)Exhibit\s+\([a-z]\)\(\d+\)',           ''),
    (r'(?i)CONFIDENTIAL\s*[-–]\s*DO NOT DISTRIBUTE', ''),
    (r'(?i)Draft\s+v\d+\.?\d*',                  ''),
    (r'(?i)EXECUTION COPY',                       ''),
    (r'\[SIGNATURE PAGE FOLLOWS\]',               ''),
    (r'\[INTENTIONALLY LEFT BLANK\]',             ''),
    (r'(?i)initial[s]?\s*[:\-]?\s*_{2,}',        ''),
    (r'_{3,}',                                    ''),
    (r'\s{2,}',                                   ' '),
    (r'\n{3,}',                                   '\n\n'),
]

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_nda_text(pdf_path):
    text_parts = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = str(page.get_text("text")).strip()
            if text:
                text_parts.append(text)

    # ── Gap 3 — no page separator ─────────────────
    full_text = "\n".join(text_parts)

    # ── Gap 4 — no unicode normalization ──────────
    full_text = full_text.replace("\u2014", "—")   # em dash
    full_text = full_text.replace("\u2013", "–")   # en dash
    full_text = full_text.replace("\ufb01", "fi")  # fi ligature
    full_text = full_text.replace("\ufb02", "fl")  # fl ligature
    full_text = full_text.replace("\xa0",   " ")   # non-breaking space

    # ── Gap 5 — no noise cleanup ──────────────────
    full_text = clean_nda_text(full_text)

    print(f"✅ Extracted {len(full_text)} chars from NDA")
    return full_text.strip()


def clean_nda_text(text):
    """
    Remove common NDA PDF noise before sending to LLM.
    Safe patterns only — does not touch clause content.
    """
    SAFE_PATTERNS = [
        (r'(?i)Page\s+\d+\s+of\s+\d+',                ''),   # Page 1 of 10
        (r'(?i)CONFIDENTIAL\s*[-–]\s*DO NOT DISTRIBUTE',''),  # watermark
        (r'(?i)EXECUTION COPY',                         ''),  # execution marker
        (r'\[SIGNATURE PAGE FOLLOWS\]',                 ''),  # sig marker
        (r'\[INTENTIONALLY LEFT BLANK\]',               ''),  # blank page
        (r'(?i)initial[s]?\s*[:\-]?\s*_{3,}',           ''),  # Initials: ____
    ]

    for pattern, replacement in SAFE_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # Normalize whitespace last
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 newlines
    text = re.sub(r'[ \t]+',  ' ',   text)   # collapse spaces

    return text.strip()


def segment_nda_llm(nda_text):
    """
    Primary segmentation — sends NDA text to Claude.
    Returns structured clause JSON or None if it fails.
    """
    prompt = PromptTemplate.from_template(SEGMENT_PROMPT)
    chain  = prompt | model

    try:
        response = chain.invoke({"nda_text": nda_text})

        raw = str(response.content).strip()

        # Strip markdown fences if model added them
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$',     '', raw)

        structured = json.loads(raw)

        # Basic structure check
        if "clauses" not in structured or not structured["clauses"]:
            print("⚠️  LLM returned empty clauses")
            return None

        print(f"✅ LLM segmented into {structured.get('total_clauses', len(structured['clauses']))} clauses")
        structured.setdefault("total_clauses", len(structured["clauses"]))
        return structured

    except json.JSONDecodeError as e:
        print(f"❌ LLM JSON parse failed: {e}")
        return None

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return None


def guess_clause_type(title):
    """
    Guesses clause type from title keywords.
    Used by regex fallback since LLM isn't classifying.
    """
    title_lower = title.lower()
    for keyword, clause_type in CLAUSE_TYPES.items():
        if keyword in title_lower:
            return clause_type
    return "general"


def segment_nda_regex(nda_text):
    """
    Fallback segmentation using regex.
    Used only when LLM segmentation fails.
    Less accurate — does not classify types precisely.
    """
    print("⚠️  Using regex fallback segmentation...")

    matches = list(header_pattern.finditer(nda_text))

    if not matches:
        print("❌ Regex found no clause headers")
        return {
            "error"        : "No headers found",
            "raw_text"     : nda_text,
            "segmented_by" : "regex"
        }

    structured_clauses = []
    clause_counter     = 1

    # ── Capture preamble ──────────────────────────
    preamble_text = nda_text[:matches[0].start()].strip()
    if preamble_text:
        structured_clauses.append({
            "clause_number" : 0,                # 0 = preamble
            "clause_title"  : "Introduction / Parties",
            "clause_text"   : preamble_text,
            "clause_type"   : "preamble"
        })

    # ── Segment clauses ───────────────────────────
    for i, match in enumerate(matches):

        next_start = matches[i + 1].start() \
                     if (i + 1) < len(matches) else len(nda_text)

        
        # Slice body by position — not string replace
        body = nda_text[match.end():next_start].strip()

        title = match.group(1).strip()

        if not body:
            continue

        structured_clauses.append({
            "clause_number" : clause_counter,
            "clause_title"  : title,
            "clause_text"   : body,
            "clause_type"   : guess_clause_type(title)
        })

        clause_counter += 1

    if clause_counter == 1:
        print("❌ Regex fallback found headers but no bodies")
        return {
            "error"        : "No clause bodies extracted",
            "raw_text"     : nda_text,
            "segmented_by" : "regex"
        }

    # Exclude preamble from clause count
    actual_clauses = [
        c for c in structured_clauses
        if c["clause_type"] != "preamble"
    ]

    print(f"⚠️  Regex extracted {len(actual_clauses)} clauses "
        f"+ preamble (lower accuracy)")

    return {
        "nda_title"    : "NDA Document",
        "total_clauses": len(actual_clauses),
        "clauses"      : structured_clauses,   # includes preamble at index 0
        "segmented_by" : "regex"
    }

def segment_nda(nda_text):
    """
    Tries LLM first, falls back to regex if LLM fails.
    Always returns structured dict or None.
    """

    # Try LLM first
    structured = segment_nda_llm(nda_text)

    if structured:
        structured["segmented_by"] = "llm"
        return structured

    # Fallback to regex
    print("⚠️  Falling back to regex segmentation")
    structured = segment_nda_regex(nda_text)

    if structured and "clauses" in structured:
        return structured

    # Both failed
    print("❌ Both LLM and regex segmentation failed")
    return None

if __name__ == "__main__":
    pdf_path = os.path.join(BASE_DIR, "documents", "Nda_document1.pdf")
    nda_text = extract_nda_text(pdf_path)
    structured = segment_nda(nda_text)

    if not structured:
        print("❌ Segmentation failed completely")
        exit(1)

    # ── Save to data/structured_nda.json ──────────
    output_path = os.path.join(BASE_DIR, "data", "structured_nda.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved to {output_path}")

    print(f"\n── Result ───────────────────────────────────")