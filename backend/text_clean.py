import fitz
from dotenv import load_dotenv
import os
import re

load_dotenv()

pdf_path = os.getenv("pdf_path")

REPEALED = set(range(76, 124)) | set(range(239, 267))

NDA_RELEVANT = {
    "2",  "10", "11", "12",
    "13", "14", "15", "16",
    "17", "18", "19", "19A",
    "20", "21", "22", "23",
    "27", "28", "56",
    "73", "74", "75"
}

CRITICAL = {"16", "17", "27", "28", "73", "74"}

def load_full_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    #Extract text from each page and concatenate
    text_parts = []
    for page in doc:
        text = str(page.get_text("text")).strip()
        if text:
            text_parts.append(text)

    full_text = "\n".join(text_parts)

    full_text = full_text.replace("\u2014", "—")   # em dash variants
    full_text = full_text.replace("\u2013", "–")   # en dash
    full_text = full_text.replace("\ufb01", "fi")  # ligature fi
    full_text = full_text.replace("\ufb02", "fl")  # ligature fl
    
    return full_text



def remove_table_of_contents(text):
    """Strip TOC — everything before actual PRELIMINARY section."""
    match = re.search(
        r'PRELIMINARY\s*\n+\s*1\.\s+Short title',
        text
    )
    if match:
        print("✅ TOC removed")
        return text[match.start():]
    print("⚠️  TOC marker not found — check manually")
    return text


def clean_extracted_text(raw_text):
     # Step 1 — Remove page headers
    raw_text = re.sub(r'THE INDIAN CONTRACT ACT,?\s*1872\s*\n', '', raw_text)

    # Step 2 — Remove standalone page numbers
    raw_text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', raw_text)

    # Step 3 — Remove footnote block lines
    raw_text = re.sub(
        r'\n\d{1,2}\.\s+(?:Subs\.|Ins\.|Rep\.|See|Cf\.|Added|The\s|For\s|As\s)[^\n]+',
        '', raw_text, flags=re.MULTILINE
    )

    # Step 4 — Remove inline footnote superscripts
    raw_text = re.sub(r'(?<=[a-zA-Z])\d{1,2}(?=[\s,\.;\—])', '', raw_text)

    # Step 4b — Remove repealed asterisk blocks
    raw_text = re.sub(r'\n\s*\d?\*[\s\*\.]*', '', raw_text)

    # Remove inline footnote bracket markers
    raw_text = re.sub(r'\d{1,2}\[', '', raw_text)
    raw_text = re.sub(r'(?<=[a-z\.\,])\]', '', raw_text)

    # ── Step 5 — PROTECT section boundaries BEFORE line joining ──
    # 5a — Insert newline when section number is glued to previous sentence
    # e.g. "...the business. 28.Agreements" → "...the business.\n\n28.Agreements"
    raw_text = re.sub(
        r'([.!?])\s+(\d{1,3}[A-Z]?\.\s{0,4}[A-Z])',
        r'\1\n\n\2',
        raw_text
    )

    # 5b — Add double newline before any line starting with a section number
    # This prevents line joiner from eating section headers
    raw_text = re.sub(
        r'\n(\d{1,3}[A-Z]?\.\s{0,4}[A-Z"\u201c])',
        r'\n\n\1',
        raw_text
    )

    # Step 6 — Join mid-sentence line breaks
    # Now safe — section headers are protected by double newline
    # Single \n between lowercase continuation = join
    # Double \n before section header = never touched by this regex
    raw_text = re.sub(
        r'(?<![.!?:\—\-])\n(?=[a-z\(])',
        ' ', raw_text
    )

    # Step 7 — Normalize whitespace AFTER joining
    raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
    raw_text = re.sub(r'[ \t]+', ' ', raw_text)

    return raw_text.strip()

def parse_sections(text):

    text = remove_table_of_contents(text)
    text = clean_extracted_text(text)

    section_pattern = re.compile(
        r'(?m)^(\d{1,3}[A-Z]?)\.'
        r'\s{0,4}'
        r'([^\n]+?)'
        r'(?:\s*\.—|\s*—)',
    )

    all_matches = list(section_pattern.finditer(text))
    print(f"📍 Boundaries detected: {len(all_matches)}")

    if not all_matches:
        print("❌ No sections found — check input")
        return []

    sections = []

    for i, match in enumerate(all_matches):

        section_num   = match.group(1).strip()
        section_title = match.group(2).strip()

        # ── Body slicing FIRST — before any filtering ──
        # Uses all_matches so positions are always correct
        body_start = match.end()
        body_end   = all_matches[i + 1].start() if (i + 1) < len(all_matches) else len(text)
        body       = text[body_start:body_end].strip()

        # ── NOW filter repealed ────────────────────────
        try:
            num_only = int(re.sub(r'[A-Z]', '', section_num))
            if num_only in REPEALED:
                continue
        except ValueError:
            pass

        # ── Skip empty or repealed stubs ──────────────
        if not body or body.lower().startswith('rep.'):
            continue

        sections.append({
            "section" : section_num,
            "title"   : section_title,
            "body"    : body
        })

    print(f"✅ Parsed {len(sections)} valid sections")

    # ── Filter NDA relevant ────────────────────────────
    nda_sections = [
        s for s in sections
        if s["section"] in NDA_RELEVANT
    ]

    print(f"✅ NDA relevant sections: {len(nda_sections)}")

    # ── Verify critical sections ───────────────────────
    found   = {s["section"] for s in nda_sections}
    missing = CRITICAL - found

    if missing:
        print(f"⚠️  Missing critical sections: {missing}")
        for sec in missing:
            nearby = [
                m.group(0) for m in all_matches
                if m.group(1).startswith(sec[0])
            ][:3]
            print(f"   Near matches for S.{sec}: {nearby}")
    else:
        print(f"✅ All critical sections present")

    return nda_sections


if __name__ == "__main__":

    # ── Step 1: Load ──────────────────────────────────
    full_text = load_full_pdf(pdf_path)

    # ── Step 2: Parse ─────────────────────────────────
    sections = parse_sections(full_text)

    # ── Step 3: Completeness check ────────────────────
    COMPLETENESS_CHECKS = {
        "16" : ["position to dominate", "burden of proving", "fiduciary"],
        "17" : ["active concealment", "mere silence", "Illustrations"],
        "27" : ["restrained from exercising", "Exception", "good-will"],
        "28" : ["restricted absolutely", "arbitration", "Exception"],
        "73" : ["naturally arose", "remote and indirect", "Illustrations"],
        "74" : ["whether or not actual damage", "bail-bond", "penalty"],
    }

    print("\n── Completeness Report ──────────────────────")
    all_good = True

    for sec_num, required_phrases in COMPLETENESS_CHECKS.items():
        sec = next(
            (s for s in sections if s["section"] == sec_num),
            None
        )

        if not sec:
            print(f"❌ S.{sec_num} — NOT FOUND")
            all_good = False
            continue

        # NEW — whitespace insensitive matching
        missing_phrases = [
            p for p in required_phrases
            if not re.search(
                re.sub(r'\s+', r'\\s+', p),   # "restrained from exercising"
                sec['body'],                   # → "restrained from\s+exercising"
                re.IGNORECASE                  # matches 1 or more spaces/newlines
            )
        ]

        if missing_phrases:
            print(f"⚠️  S.{sec_num} — incomplete, missing: {missing_phrases}")
            all_good = False
        else:
            print(f"✅ S.{sec_num} — complete ({len(sec['body'])} chars)")

    # ── Step 4: Inspect specific section ──────────────
    for target in ["27", "74"]:
        sec = next(
            (s for s in sections if s["section"] == target),
            None
        )
        if sec:
            print(f"\n--- S.{target} FULL BODY ({len(sec['body'])} chars) ---")
            print(sec['body'])
            print(f"--- END S.{target} ---")
        else:
            print(f"❌ S.{target} not found")

    # ── Step 5: Final status ───────────────────────────
    if all_good:
        print("\n✅ All sections complete — ready for merge and enrich stage")
    else:
        print("\n⚠️  Fix incomplete sections before proceeding")