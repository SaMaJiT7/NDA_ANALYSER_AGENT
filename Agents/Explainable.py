import os
import re
import json
import logging
from typing import Optional
from dotenv import load_dotenv

from groq import Groq


load_dotenv()
# ── Setup logging ─────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()



COT_AND_EXPLAIN_PROMPT = load_prompt("prompts/COT_and_Explain_Prompt.md")


_groq_client = None          # global — lowercase g

def load_Groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client

# ── Retrival Explainability ─────────────────────────────
def retrival_explain(retrieved_sections: list[dict]) -> dict:
    """
    Explains WHY certain ICA sections were retrieved.

    Uses Qdrant similarity scores to compute:
    - Which section ranked first and by what margin
    - Score delta between rank 1 and rank 2 (confidence gap)
    - Human-readable rank explanation
    """

    if not retrieved_sections:
        return {
            "top_section"     : None,
            "top_score"       : 0.0,
            "score_delta"     : 0.0,
            "confidence"      : "LOW",
            "rank_explanation": "No ICA sections retrieved — uncertain mapping",
            "all_sections"    : []
        }

    # ── Sort by score descending ──────────────────
    ranked = sorted(retrieved_sections, key=lambda x: x.get("score", 0), reverse=True)

    top = ranked[0]
    top_score = top.get("score", 0.0)
    top_sec = top.get("section", "Unknown")

    if len(ranked) > 1:
        second_score = ranked[1].get("score", 0.0)
        delta        = round(top_score - second_score, 4)
    else:
        delta = round(top_score, 4)
    
    # ── Confidence categorization ───────────────
    if top_score >= 0.85:
        confidence = "HIGH"
    elif top_score >= 0.75:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── Rank explanation ──────────────────────────
    if delta >= 0.08:
        margin_desc = f"clear margin of {delta}"
    elif delta >= 0.04:
        margin_desc = f"moderate margin of {delta}"
    else:
        margin_desc = f"narrow margin of {delta} — ambiguous mapping"

    rank_explanation = (
        f"S.{top_sec} ranked 1st (score: {top_score:.4f}) "
        f"with {margin_desc} over next section"
    )

    # ── Build all sections list ───────────────────
    all_sections = [
        {
            "section"      : f"S.{r.get('section', '?')}",
            "score"        : round(r.get("score", 0.0), 4),
            "rank"         : idx + 1,
            "legal_concept": r.get("legal_concept", ""),
            "employee_risk": r.get("employee_risk", "")
        }
        for idx, r in enumerate(ranked)
    ]

    return {
        "top_section"     : f"S.{top_sec}",
        "top_score"       : round(top_score, 4),
        "score_delta"     : delta,
        "confidence"      : confidence,
        "rank_explanation": rank_explanation,
        "all_sections"    : all_sections
    }


def keyword_explain(clause_text : str, retrieved_sections: list[dict]) -> dict:
    """
    Explains which keywords in the clause text
    matched the ICA section trigger keywords.

    Uses token overlap between clause text and
    each retrieved section's trigger_keywords list.
    """

    if not retrieved_sections or not clause_text:
        return {
            "matched_keywords": [],
            "overlap_score"   : 0.0,
            "per_section"     : [],
            "explanation"     : "No keywords to match"
        }
    
    clause = clause_text.lower()
    

    per_section = []
    all_matched_keywords = set()

    for section in retrieved_sections:
        trigger_keywords = section.get("trigger_keywords", [])
        section_id = section.get("section", "Unknown")


        if not trigger_keywords:
            continue

        matched = [
            keyword for keyword in trigger_keywords
            if keyword.lower() in clause
        ]

        # ── Overlap score for this section ────────
        overlap = round(len(matched) / len(trigger_keywords), 3) if trigger_keywords else 0.0

        all_matched_keywords.update(matched)

        per_section.append({
            "section"          : section_id,
            "trigger_keywords" : trigger_keywords,
            "matched_keywords" : matched,
            "total_keywords"   : len(trigger_keywords),
            "matched_count"    : len(matched),
            "overlap_score"    : overlap
        })
        
    # ── Overall overlap score ─────────────────────
    if per_section:
        overall_overlap = round(
            sum(s["overlap_score"] for s in per_section) / len(per_section), 3
        )
    else:
        overall_overlap = 0.0

    # Human explanation
    if all_matched_keywords:
        kw_list     = ", ".join(f"'{k}'" for k in sorted(all_matched_keywords)[:5])
        explanation = (
            f"Keywords {kw_list} found in clause text — "
            f"triggered retrieval of {len(per_section)} ICA section(s)"
        )
    else:
        explanation = "No trigger keywords matched — retrieval driven by semantic similarity only"

    return {
        "matched_keywords": sorted(all_matched_keywords),
        "overlap_score"   : overall_overlap,
        "per_section"     : per_section,
        "explanation"     : explanation
    }

def _extract_think_block(raw: str) -> str:
    """Extracts content inside <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_json_after_think(raw: str) -> dict:
    """Extracts and parses JSON after the </think> block."""
    json_text = raw.split("</think>")[-1].strip() \
                if "</think>" in raw else raw

    # Strip markdown fences if present
    json_clean = re.sub(r"```json|```", "", json_text).strip()

    try:
        return json.loads(json_clean)
    except Exception:
        return {}

def _summarise_think(think_text: str) -> list[str]:
    """
    Extracts key legal reasoning sentences from <think> block.
    Keeps sentences that mention ICA sections or risk signals.
    """
    if not think_text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', think_text)

    LEGAL_SIGNALS = [
        "s.27", "s.74", "s.28", "s.16", "s.17", "s.23", "s.73",
        "void", "voidable", "enforceable", "restraint",
        "employee", "high risk", "illegal", "indian contract act"
    ]

    key_sentences = [
        s.strip() for s in sentences
        if any(sig in s.lower() for sig in LEGAL_SIGNALS)
        and len(s.strip()) > 20
    ]

    return key_sentences[:5]

def cot_and_explain(
    clause_title  : str,
    clause_type   : str,
    clause_text   : str,
    risk_level    : str,
    retrieval_xai : dict,
    keyword_xai   : dict,
) -> dict:
    """
    Does CoT legal reasoning AND plain-language explanation in ONE Groq call.
    Returns dict with:
      cot_xai          : {think_block, think_summary, key_concern, critical_phrases, cot_risk_level, agrees_with_analyser}
      human_explanation: plain-language string
    """
    if not clause_text:
        return {
            "cot_xai": {
                "think_block"         : "",
                "think_summary"       : [],
                "key_concern"         : "No clause text provided",
                "critical_phrases"    : [],
                "cot_risk_level"      : risk_level,
                "agrees_with_analyser": True
            },
            "human_explanation": f"This clause was flagged {risk_level} risk. No text was available for deeper analysis."
        }
    
    client = load_Groq_client()

    prompt = (COT_AND_EXPLAIN_PROMPT
        .replace("{clause_title}",          clause_title)
        .replace("{clause_type}",           clause_type)
        .replace("{risk_level}",            risk_level)
        .replace("{clause_text}",           clause_text)
        .replace("{retrieval_explanation}", retrieval_xai.get("rank_explanation", "N/A"))
        .replace("{keyword_explanation}",   keyword_xai.get("explanation", "N/A"))
    )

    try:
        response = client.chat.completions.create(
            model            = "qwen/qwen3-32b",
            messages         = [{"role": "user", "content": prompt}],
            max_tokens       = 1500,
            temperature      = 0.6,
            reasoning_effort = "default"
        )

        raw = str(response.choices[0].message.content)
        think = _extract_think_block(raw)
        parsed = _extract_json_after_think(raw)
        cot_risk = parsed.get("cot_risk_level", risk_level)

        cot_xai = {
            "think_block"         : think,
            "think_summary"       : _summarise_think(think),
            "key_concern"         : parsed.get("key_concern", ""),
            "critical_phrases"    : parsed.get("critical_phrases", []),
            "cot_risk_level"      : cot_risk,
            "agrees_with_analyser": cot_risk == risk_level
        }
        
        human_explanation = parsed.get("human_explanation", "").strip()
        if not human_explanation:
            human_explanation = (
                f"This clause was flagged {risk_level} risk. "
                f"{retrieval_xai.get('rank_explanation', '')} "
                f"{keyword_xai.get('explanation', '')}"
            )

        return {
            "cot_xai"          : cot_xai,
            "human_explanation": human_explanation
        }
    
    except Exception as e:
        logger.warning(f"     ↳ ⚠️  CoT+Explain failed: {e}")
        return {
            "cot_xai": {
                "think_block"         : "",
                "think_summary"       : [],
                "key_concern"         : f"Analysis unavailable: {e}",
                "critical_phrases"    : [],
                "cot_risk_level"      : risk_level,
                "agrees_with_analyser": True
            },
            "human_explanation": (
                f"This clause was flagged {risk_level} risk. "
                f"{retrieval_xai.get('rank_explanation', '')} "
                f"{keyword_xai.get('explanation', '')}"
            )
        }


def run_xai(risk_report: dict) -> dict:
    """
    Runs full XAI pipeline on all assessed clauses.

    For each clause:
      1. retrival_explain()   — Qdrant score analysis      (no API call)
      2. keyword_explain()    — token overlap               (no API call)
      3. cot_and_explain()    — CoT + human explanation     (1 Groq call)

    Optimisations:
      - LOW risk clauses skipped from Groq call (saves ~70-80% quota)
      - 2 Groq calls per clause → 1 Groq call per clause
      - Rate limit buffer only applied when Groq is called

    Returns xai_report dict.
    """
    import time

    logger.info("\n── XAI Agent Starting ───────────────────────")

    clause_reports = risk_report.get("clause_reports", [])
    xai_clauses    = []
    skipped_count  = 0

    for clause in clause_reports:

        # ── Skip preamble and unassessed clauses ──
        if clause.get("skipped"):
            skipped_count += 1   # ← add this
            continue

        assessment = clause.get("assessment")
        if not assessment:
            skipped_count += 1   # ← add this
            continue

        clause_number = clause.get("clause_number", "?")
        clause_title  = clause.get("clause_title", "Untitled")
        clause_type   = clause.get("clause_type", "general")
        clause_text   = clause.get("clause_text", "")
        risk_level    = assessment.get("risk_level", "LOW")
        retrieved     = clause.get("retrieved_sections", [])

        logger.info(
            f"  🔍 XAI — Clause {clause_number} "
            f"— {clause_title} [{risk_level}]"
        )

        # ── Fix 1 — Warn if clause_text missing ───
        # explain_keywords() and explain_cot() need this
        if not clause_text:
            logger.warning(
                f"     ↳ ⚠️  clause_text empty for clause {clause_number} "
                f"— keyword and CoT XAI will be limited"
            )

        # ── Component 1 — Retrieval XAI ──────────
        # Pure math — no API call, no rate limit
        retrieval_xai = retrival_explain(retrieved)
        logger.info(
            f"     ↳ Retrieval : {retrieval_xai['top_section']} "
            f"score={retrieval_xai['top_score']} "
            f"delta={retrieval_xai['score_delta']} "
            f"confidence={retrieval_xai['confidence']}"
        )

        # ── Component 2 — Keyword XAI ─────────────
        # Token overlap — no API call, no rate limit
        keyword_xai = keyword_explain(clause_text, retrieved)
        logger.info(
            f"     ↳ Keywords  : {keyword_xai['matched_keywords'][:5]} "
            f"overlap={keyword_xai['overlap_score']}"
        )
        clause_text_safe = clause_text.replace("{", "{{").replace("}", "}}")

        # ── Component 3 & 4 — CoT + Explanation ──────────────────────────
        # LOW risk clauses: skip Groq call → use lightweight fallback
        # MEDIUM/HIGH risk:  1 Groq call via cot_and_explain()
        if risk_level == "LOW":
            logger.info(f"     ↳ LOW risk — skipping Groq CoT (quota optimisation)")
            cot_xai = {
                "think_block"         : "",
                "think_summary"       : [],
                "key_concern"         : "Standard clause — low legal risk under ICA",
                "critical_phrases"    : [],
                "cot_risk_level"      : "LOW",
                "agrees_with_analyser": True
            }
            human_explanation = (
                f"This is a standard {clause_type} clause rated LOW risk. "
                f"{retrieval_xai.get('rank_explanation', '')} "
                f"No immediate action required, but review before signing."
            )
        else:
            result = cot_and_explain(
                clause_title  = clause_title,
                clause_type   = clause_type,
                clause_text   = clause_text_safe,
                risk_level    = risk_level,
                retrieval_xai = retrieval_xai,
                keyword_xai   = keyword_xai,
            )
            cot_xai = result["cot_xai"]
            human_explanation = result["human_explanation"]

            # ── Truncate think_block for storage ──────
            if len(cot_xai.get("think_block", "")) > 500:
                cot_xai["think_block"] = cot_xai["think_block"][:500] + "...[truncated]"

            # ── Log CoT result ────────────────────────
            logger.info(
                f"     ↳ CoT       : agrees={cot_xai['agrees_with_analyser']} "
                f"| concern={cot_xai.get('key_concern', '')[:60]}..."
            )
            if not cot_xai["agrees_with_analyser"]:
                logger.warning(
                    f"     ↳ ⚠️  Risk mismatch — "
                    f"Analyser={risk_level} | "
                    f"Qwen3={cot_xai['cot_risk_level']}"
                )

            # ── Rate limit buffer ─────────────────
            time.sleep(2)

        logger.info(f"     ↳ Explanation: {human_explanation[:80]}...")

        xai_clauses.append({
            "clause_number"    : clause_number,
            "clause_title"     : clause_title,
            "clause_type"      : clause_type,
            "risk_level"       : risk_level,
            "retrieval_xai"    : retrieval_xai,
            "keyword_xai"      : keyword_xai,
            "cot_xai"          : cot_xai,
            "human_explanation": human_explanation
        })

    # ── Build XAI report ──────────────────────────
    xai_report = {
        "nda_title"      : risk_report.get("nda_title"),
        "company_name"   : risk_report.get("company_name"),
        "risk_score"     : risk_report.get("risk_score"),
        "risk_label"     : risk_report.get("risk_label"),
        "xai_clauses"    : xai_clauses,
        "total_explained": len(xai_clauses),
        "total_skipped"  : skipped_count
    }

    logger.info(
        f"\n✅ XAI complete — "
        f"{len(xai_clauses)} explained | "
        f"{skipped_count} skipped"
    )
    return xai_report


# ── Save XAI report ───────────────────────────────
def save_xai_report(
    xai_report: dict,
    path: str = "xai_report.json"
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(xai_report, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ XAI report saved to {path}")


# ── Main ──────────────────────────────────────────
if __name__ == "__main__":

    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

    # ── Load risk_report.json directly if available ───
    risk_report_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "risk_report.json"
    )

    if os.path.exists(risk_report_path):
        logger.info(f"✅ Loading existing risk_report from {risk_report_path}")
        with open(risk_report_path, "r", encoding="utf-8") as f:
            risk_report = json.load(f)

    else:
        # ── Fallback: run full pipeline ───────────────
        logger.info("ℹ️  risk_report.json not found — running full pipeline")

        from Agents.Analyser  import run_analyser
        from Agents.Validator import validate_report

        structured_nda_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "structured_nda.json"
        )

        if not os.path.exists(structured_nda_path):
            logger.error("❌ structured_nda.json not found — run orchestrator first")
            sys.exit(1)

        with open(structured_nda_path, "r", encoding="utf-8") as f:
            structured_nda = json.load(f)

        logger.info("── Running Analyser ─────────────────────────")
        risk_report = run_analyser(structured_nda)

        logger.info("\n── Running Validator ────────────────────────")
        validation = validate_report(risk_report)

        if validation.get("valid") == "Rejected":
            logger.error("❌ Validator rejected report — fix analyser first")
            sys.exit(1)

        if validation.get("valid") == "Needs Revision":
            logger.warning(
                f"⚠️  Validator flagged issues but proceeding to XAI\n"
                f"   Feedback: {validation.get('feedback')}"
            )

    logger.info("\n── Running XAI Agent ────────────────────────")
    xai_report = run_xai(risk_report)
    save_xai_report(xai_report, path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "xai_report.json"
    ))

    # ── Summary ───────────────────────────────────
    print(f"\n── XAI Summary ──────────────────────────────")
    print(f"  NDA          : {xai_report['nda_title']}")
    print(f"  Risk Score   : {xai_report['risk_score']}/100")
    print(f"  Risk Label   : {xai_report['risk_label']}")
    print(f"  Explained    : {xai_report['total_explained']} clauses")
    print()

    for clause in xai_report["xai_clauses"]:
        print(f"  Clause {clause['clause_number']} — {clause['clause_title']}")
        print(f"    Risk       : {clause['risk_level']}")
        print(f"    Top Section: {clause['retrieval_xai']['top_section']} "
              f"(score: {clause['retrieval_xai']['top_score']}, "
              f"confidence: {clause['retrieval_xai']['confidence']})")
        print(f"    Keywords   : {clause['keyword_xai']['matched_keywords'][:5]}")
        print(f"    CoT Agrees : {clause['cot_xai']['agrees_with_analyser']} "
              f"| Risk: {clause['cot_xai']['cot_risk_level']}")
        print(f"    Key Concern: {clause['cot_xai']['key_concern']}")
        print(f"    Explanation: {clause['human_explanation']}")
        print()