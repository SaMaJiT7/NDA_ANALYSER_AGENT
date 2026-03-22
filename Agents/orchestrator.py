import os
import sys
import json
import logging
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END

from tools.segment import run_segmentation_pipeline
from Agents.Analyser import run_analyser, save_report

load_dotenv()

# ── Setup logging ─────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3

class NDAState(TypedDict):
    # ── Input ─────────────────────────────────────
    pdf_path        : str

    # ── Pipeline data ─────────────────────────────
    structured_nda  : Optional[dict]
    risk_report     : Optional[dict]
    validation      : Optional[dict]
    xai_report      : Optional[dict]
    final_response  : Optional[str]

    # ── output data ─────────────────────────────
    segment_output_path : Optional[str]
    

    # ── Control flow ──────────────────────────────
    retry_count     : int
    error           : Optional[str]
    feedback        : Optional[str]
    failed_clauses  : Optional[list]


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

