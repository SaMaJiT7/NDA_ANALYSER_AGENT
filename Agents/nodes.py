from Agents.state import NDAState
from dotenv import load_dotenv
import logging

from tools.segment import run_segmentation_pipeline


load_dotenv()

# ── Setup logging ─────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def segment_node(state: NDAState) -> NDAState:
    """
    Node 1 — Segments NDA PDF into structured clauses.
    Input  : pdf_path
    Output : structured_nda
    """
    logger.info("Running the Segmentation Node...")
    try:
        pdf_path = state["pdf_path"]
        result = run_segmentation_pipeline(pdf_path)
        if not result:
            return {**state, "error": "Segmentation failed"}
        structured_nda, structured_path = result
        if not structured_nda or not structured_nda.get("clauses"):
            return {
                **state,
                "error": "Segmenter returned no clauses"
            }

        logger.info(
            f"  ✅ Segmented — "
            f"{structured_nda.get('total_clauses', 0)} clauses found"
            f"  📄 Saved to: {structured_path}"
        )

        return {
            **state,
            "segment_output_path": structured_path,
            "structured_nda": structured_nda,
            "error"         : None
        }

    except Exception as e:
        logger.error(f"  ❌ Segmenter failed: {e}")
        return {**state, "error": f"Segmenter failed: {e}"}


def analyse_node(state: NDAState) -> NDAState:
    """
    Node 2 — Analyses each clause using RAG + ICA vector DB.
    Input  : structured_nda (+ optional feedback on retry)
    Output : risk_report
    """
    from Agents.Analyser import run_analyser

    retry = state.get("retry_count", 0)

    # ── Check max retries before running ─────────
    if retry >= MAX_RETRIES:
        logger.warning(
            f"\n── [Node 2] Analyser ────────────────────────\n"
            f"  ⚠️  Max retries ({MAX_RETRIES}) reached — skipping re-analysis\n"
            f"  Proceeding with best available report"
        )
        return state    # ← return as-is, validate_node will route to responder

    logger.info(
        f"\n── [Node 2] Analyser "
        f"{'(retry ' + str(retry) + '/' + str(MAX_RETRIES) + ')' if retry > 0 else ''}"
        f" ───────────────────"
    )
    
    try:
        structured_nda = state["structured_nda"]
        if not structured_nda:
            return {**state, "error": "structured_nda missing from state"}
        feedback = state.get("feedback", None)
        failed_clauses = state.get("failed_clauses")

        risk_report = run_analyser(
            structured_nda,
            feedback       = feedback,
            failed_clauses = failed_clauses
        )

        if not risk_report:
            return {**state, "error": "Analyser returned empty report"}

        logger.info(
            f"  ✅ Analysed — "
            f"Score: {risk_report.get('risk_score')}/100 | "
            f"Label: {risk_report.get('risk_label')}"
        )

        return {
            **state,
            "risk_report"   : risk_report,
            "feedback"      : None,        # ← clear after use
            "failed_clauses": None,
            "error"         : None
        }

    except Exception as e:
        logger.error(f"  ❌ Analyser failed: {e}")
        return {**state, "error": f"Analyser failed: {e}"}

    
def validate_node(state: NDAState) -> NDAState:
    """
    Node 3 - Validates the risk report and generates the feedback for re-analysis if needed.
    Input  : risk_report
    Output : validation
    """
    from Agents.Validator import validate_report

    risk_report = state.get("risk_report")

    if not risk_report:
        return {**state, "error": "risk_report missing from state"}
    
    try:
        validation = validate_report(risk_report)

        status = validation.get("valid", "Rejected")
        icon   = "✅" if status == "Approved" else \
                 "⚠️ " if status == "Needs Revision" else "❌"

        logger.info(
            f"  {icon} Status     : {status}\n"
            f"  Confidence : {validation.get('confidence_score')}\n"
            f"  Citation   : {validation.get('citation_valid')}"
        )

        return {
            **state,
            "validation"    : validation,
            "feedback"      : validation.get("feedback"),
            "failed_clauses": validation.get("failed_clauses"),
            "error"         : None
        }

    except Exception as e:
        logger.error(f"  ❌ Validator failed: {e}")
        return {**state, "error": f"Validator failed: {e}"}



def explainable_node(state: NDAState) -> NDAState:

    """
    Node 4 - Generates XAI report for the risk report.
    Input  : risk_report
    Output : xai_report
    """
    from Agents.Explainable import run_xai

    risk_report = state.get("risk_report")

    if not risk_report:
        return {**state, "error": "risk_report missing from state"}
    
    try:
        xai_report = run_xai(risk_report)
        
        logger.info(
            f"  ✅ XAI Report generated — "
            f"Key clauses: {xai_report.get('key_clauses', [])[:3]}"
        )
        
        return {
            **state,
            "xai_report": xai_report,
            "error": None
        }
    except Exception as e:
        logger.error(f"  ❌ XAI Report generation failed: {e}")
        return {**state, "error": f"XAI Report generation failed: {e}"}
    

def respond_node(state: NDAState) -> NDAState:
    """
    Node 5 — Final responder.
    Assembles risk_report + xai_report into a user-facing summary.

    Two entry paths:
      1. Happy path  : after XAI   → has risk_report + xai_report
      2. Fallback    : after retry cap / error → risk_report only
    """
    import json, os

    logger.info("\n── [Node 5] Responder ───────────────────────")

    risk_report = state.get("risk_report")
    xai_report  = state.get("xai_report")

    # ── Guard — nothing to report ─────────────────
    if not risk_report:
        logger.error("  ❌ No risk_report in state — cannot generate response")
        return {
            **state,
            "final_response": "Analysis could not be completed. No risk report was generated.",
            "error": "respond_node: risk_report missing"
        }

    # ── Build header ──────────────────────────────
    lines = [
        "═" * 60,
        f"  NDA RISK ANALYSIS REPORT",
        "═" * 60,
        f"  NDA Title      : {risk_report.get('nda_title', 'Unknown')}",
        f"  Company        : {risk_report.get('company_name', 'Unknown')}",
        f"  Risk Score     : {risk_report.get('risk_score', 'N/A')}/100",
        f"  Risk Label     : {risk_report.get('risk_label', 'N/A')}",
        f"  Recommendation : {risk_report.get('recommendation', 'N/A')}",
        "",
        f"  Clauses Assessed : {risk_report.get('total_assessed', 0)}",
        f"  Clauses Skipped  : {risk_report.get('total_skipped', 0)}",
        "─" * 60,
    ]

    # ── Clause-by-clause detail ───────────────────
    xai_map = {}
    if xai_report:
        for xc in xai_report.get("xai_clauses", []):
            xai_map[xc.get("clause_number")] = xc

    for clause in risk_report.get("clause_reports", []):
        assessment = clause.get("assessment")
        if not assessment or clause.get("skipped"):
            continue

        num   = clause.get("clause_number", "?")
        title = clause.get("clause_title", "Untitled")
        risk  = assessment.get("risk_level", "LOW")

        lines.append(f"\n  Clause {num} — {title}")
        lines.append(f"    Risk Level     : {risk}")
        lines.append(f"    Void           : {'YES' if assessment.get('is_void') else 'NO'}")
        lines.append(f"    Sections       : {assessment.get('violated_sections', [])}")
        lines.append(f"    Red Flags      : {assessment.get('red_flags', [])}")
        lines.append(f"    Negotiate      : {assessment.get('negotiation_points', [])}")

        # ── Add XAI explanation if available ──────
        xai = xai_map.get(num)
        if xai:
            lines.append(f"    Explanation    : {xai.get('human_explanation', 'N/A')}")
            cot = xai.get("cot_xai", {})
            if cot.get("key_concern"):
                lines.append(f"    Key Concern    : {cot['key_concern']}")
            if not cot.get("agrees_with_analyser"):
                lines.append(
                    f"    ⚠️  CoT Risk    : {cot.get('cot_risk_level')} "
                    f"(differs from analyser: {risk})"
                )
        else:
            lines.append(f"    Reasoning      : {assessment.get('reasoning', 'N/A')}")

        lines.append("    " + "─" * 56)

    # ── Footer ────────────────────────────────────
    breakdown = risk_report.get("breakdown", {})
    lines.append(f"\n  BREAKDOWN")
    lines.append(f"    HIGH   clauses : {breakdown.get('high_clauses', 0)}")
    lines.append(f"    MEDIUM clauses : {breakdown.get('medium_clauses', 0)}")
    lines.append(f"    LOW    clauses : {breakdown.get('low_clauses', 0)}")
    if breakdown.get("void_clauses"):
        lines.append(f"    VOID   clauses : {breakdown['void_clauses']}")
    lines.append("═" * 60)

    final_response = "\n".join(lines)

    # ── Save combined report JSON ─────────────────
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        combined = {
            "risk_report": risk_report,
            "xai_report" : xai_report,
            "final_response": final_response
        }
        output_path = os.path.join(base_dir, "data", "final_report.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✅ Final report saved to {output_path}")
    except Exception as e:
        logger.warning(f"  ⚠️  Could not save final report: {e}")

    logger.info(f"  ✅ Response generated ({len(lines)} lines)")

    return {
        **state,
        "final_response": final_response,
        "error": None
    }