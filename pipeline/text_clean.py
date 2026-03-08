import fitz
from dotenv import load_dotenv
import os
import re
import json

load_dotenv()

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

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

def check_lengths(sections):
    print("\n── Length Report ────────────────────────────")
    needs_split = []

    for sec in sections:
        chars           = len(sec['body'])
        estimated_tokens = chars // 4      # 1 token ≈ 4 chars

        if estimated_tokens > 512:
            status = "⚠️  SPLIT NEEDED"
            needs_split.append(sec['section'])
        elif estimated_tokens > 400:
            status = "🟡 BORDERLINE"
        else:
            status = "✅ OK"

        print(f"{status} | S.{sec['section']:<4} | "
              f"{chars:>5} chars | ~{estimated_tokens:>4} tokens | "
              f"{sec['title'][:40]}")

    print(f"\nSections needing split: {needs_split}")
    return needs_split

def deduplicate_sections(sections):
    
    PREAMBLE_STARTS = (
        "it has been",
        "for the statement",
        "this act has been",
        "the act has been",
        "act no.",
        "it extends",
        "whereas",
    )

    LEGAL_STARTS = (
        "in this act",        # S.2
        "every person",       # S.11
        "all agreements",     # S.10
        "a contract",
        "when",
        "where",
        '"',                  # defined terms start with quote
    )

    seen   = {}
    result = []

    for sec in sections:
        num = sec['section']

        if num not in seen:
            seen[num] = len(result)
            result.append(sec)
        else:
            existing_idx  = seen[num]
            existing_body = result[existing_idx]['body'].lower().strip()
            new_body      = sec['body'].lower().strip()

            existing_is_preamble = any(
                existing_body.startswith(p) for p in PREAMBLE_STARTS
            )
            new_is_preamble = any(
                new_body.startswith(p) for p in PREAMBLE_STARTS
            )

            # Prefer legal content over preamble
            if existing_is_preamble and not new_is_preamble:
                result[existing_idx] = sec
                print(f"  🔄 S.{num} — replaced preamble with legal text")
            elif not existing_is_preamble and new_is_preamble:
                print(f"  ✅ S.{num} — kept legal text, discarded preamble")
            else:
                # Both look legal — keep first occurrence
                # First match is always closer to actual section position
                print(f"  ℹ️  S.{num} — kept first occurrence")

    duplicates_found = len(sections) - len(result)
    if duplicates_found:
        print(f"✅ Deduplication: removed {duplicates_found} duplicate(s)")

    return result

def clean_section_body(section):
    """
    Post-process individual section bodies to remove
    content that bled in from repealed sections.
    """
    body = section['body']

    # Cut off at repealed section markers
    # These patterns indicate start of repealed content
    TRUNCATE_AT = [
        r'\n\s*\[CHAPTER VII',
        r'\n\s*76\.\s+\[',
        r'\n\s*\d{2,3}\.\s+\[.*?Rep\.',     # "76. [Goods defined.] Rep."
        r'\nCHAPTER VIII\s*\nOF INDEMNITY',  # next real chapter
    ]

    for pattern in TRUNCATE_AT:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            body = body[:match.start()].strip()
            print(f"  ✂️  S.{section['section']} — truncated at repealed content")
            break

    return {**section, "body": body}

def split_section(section, max_chars=1800):
    body = section['body']

    if len(body) <= max_chars:
        return [{
            **section,
            "chunk_id"    : f"S{section['section']}_1",
            "is_split"    : False,
            "total_chunks": 1
        }]

    # ── Natural legal boundaries ──────────────────────
    split_markers = re.compile(
        r'\n(?='
        r'Illustration[s]?[\s\n]'   # Illustrations block start
        r'|Exception\s+\d'          # Exception 1, Exception 2
        r'|Exception\.—'            # Exception.—
        r'|\(\d+\)\s+[A-Z]'         # (1) Capital letter — subsection
        r'|Explanation\.—'          # Explanation.—
        r'|\([a-r]\)'               # (a)(b)...(r) — illustration letters
        r')',
        re.IGNORECASE
    )

    parts = split_markers.split(body)

    # Fallback to paragraph split if no markers found
    if len(parts) <= 1:
        parts = re.split(r'\n{2,}', body)

    # ── Group parts into chunks under max_chars ───────
    chunks    = []
    current   = ""
    chunk_idx = 1

    for part in parts:
        # If single part itself exceeds limit — force split at sentences
        if len(part) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', part)
            for sentence in sentences:
                if len(current) + len(sentence) <= max_chars:
                    current += sentence + " "
                else:
                    if current.strip():
                        chunks.append({
                            "section"  : section['section'],
                            "title"    : section['title'],
                            "body"     : current.strip(),
                            "chunk_id" : f"S{section['section']}_{chunk_idx}",
                            "is_split" : True
                        })
                        chunk_idx += 1
                    current = sentence + " "
        else:
            if len(current) + len(part) <= max_chars:
                current += part + "\n\n"
            else:
                if current.strip():
                    chunks.append({
                        "section"  : section['section'],
                        "title"    : section['title'],
                        "body"     : current.strip(),
                        "chunk_id" : f"S{section['section']}_{chunk_idx}",
                        "is_split" : True
                    })
                    chunk_idx += 1
                current = part + "\n\n"

    # Save last chunk
    if current.strip():
        chunks.append({
            "section"  : section['section'],
            "title"    : section['title'],
            "body"     : current.strip(),
            "chunk_id" : f"S{section['section']}_{chunk_idx}",
            "is_split" : True
        })

    # Add total_chunks count
    for chunk in chunks:
        chunk["total_chunks"] = len(chunks)

    return chunks


def prepare_all_chunks(sections):
    """
    Process all sections — split long ones, keep short ones as-is.
    Returns flat list ready for embedding.
    """
    all_chunks = []

    print("\n── Splitting Report ─────────────────────────")

    for sec in sections:
        result = split_section(sec)

        if len(result) > 1:
            print(f"✂️  S.{sec['section']} → "
                  f"{len(result)} chunks "
                  f"({len(sec['body'])} chars total)")
        else:
            print(f"✅ S.{sec['section']} → "
                  f"1 chunk "
                  f"({len(sec['body'])} chars)")

        all_chunks.extend(result)

    print(f"\n── Summary ──────────────────────────────────")
    print(f"Input sections : {len(sections)}")
    print(f"Output chunks  : {len(all_chunks)}")

    return all_chunks

def main():

    # ── Step 1: Load ──────────────────────────────────
    full_text = load_full_pdf(pdf_path)

    # ── Step 2: Parse ─────────────────────────────────
    sections = parse_sections(full_text)

    sections = deduplicate_sections(sections)

    sections = [clean_section_body(s) for s in sections]

    # ── Step 3: Check lengths ─────────────────────────
    needs_split = check_lengths(sections)
    
    
    chunks = prepare_all_chunks(sections)

    output_path = os.path.join(DATA_DIR, "Chunks.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Saved {len(chunks)} chunks to {output_path}")

    print("\n── Final Chunks ─────────────────────────────")
    for chunk in chunks:
        print(f"  {chunk['chunk_id']:<12} | "
            f"{len(chunk['body']):>5} chars | "
            f"~{len(chunk['body'])//4:>4} tokens")

    # Verify nothing exceeds limit
    over_limit = [
        c for c in chunks
        if len(c['body']) // 4 > 512
    ]

    if over_limit:
        print(f"\n⚠️  Still over limit: {[c['chunk_id'] for c in over_limit]}")
    else:
        print(f"\n✅ All {len(chunks)} chunks within token limit")
        print(f"✅ Ready for embedding")


if __name__ == "__main__":
    main()