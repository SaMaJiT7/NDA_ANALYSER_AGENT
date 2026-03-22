import os
import sys
import json
import logging
from typing import TypedDict, Optional, Literal

from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END

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

    # ── Control flow ──────────────────────────────
    retry_count     : int
    error           : Optional[str]
    feedback        : Optional[str]
    failed_clauses  : Optional[list]


# ── Node: Analyser ────────────────────────────────
def analyser_node(state: NDAState) -> NDAState:
    """
    Runs the analyser agent.

    On the first pass all clauses are assessed from scratch.
    On re-runs (after a validator 'Needs Revision') only the clauses
    listed in ``state['failed_clauses']`` are re-assessed; the rest are
    carried forward from the previous ``state['risk_report']``.
    """
    from Agents.Analyser import run_analyser

    structured_nda = state.get("structured_nda")
    if not structured_nda:
        return {**state, "error": "No structured NDA in state"}  # type: ignore[return-value]

    feedback        = state.get("feedback")
    failed_clauses  = state.get("failed_clauses")
    previous_report = state.get("risk_report")

    logger.info("\n── [Orchestrator] Analyser Node ─────────────")
    if failed_clauses:
        logger.info(f"  Re-running with failed_clauses={failed_clauses}")
    if feedback:
        logger.info(f"  Feedback from validator: {feedback}")

    try:
        report = run_analyser(
            structured_nda,
            feedback=feedback,
            failed_clauses=failed_clauses,
            previous_report=previous_report,
        )
        return {**state, "risk_report": report, "error": None}  # type: ignore[return-value]
    except Exception as e:
        logger.error(f"  ❌ Analyser node error: {e}")
        return {**state, "error": str(e)}  # type: ignore[return-value]


# ── Node: Validator ───────────────────────────────
def validator_node(state: NDAState) -> NDAState:
    """
    Runs the validator agent and stores the result — including any
    ``feedback`` and ``failed_clauses`` — back into the state so the
    next iteration of the analyser can use them.
    """
    from Agents.Validator import validate_report

    risk_report = state.get("risk_report")
    if not risk_report:
        return {**state, "error": "No risk report in state"}  # type: ignore[return-value]

    logger.info("\n── [Orchestrator] Validator Node ────────────")

    try:
        result = validate_report(risk_report)
        return {  # type: ignore[return-value]
            **state,
            "validation"    : result,
            "feedback"      : result.get("feedback"),
            "failed_clauses": result.get("failed_clauses"),
            "error"         : None,
        }
    except Exception as e:
        logger.error(f"  ❌ Validator node error: {e}")
        return {**state, "error": str(e)}  # type: ignore[return-value]


# ── Routing: after validator ──────────────────────
def route_after_validator(state: NDAState) -> str:
    """
    Routes control after the validator:
    - 'Approved'       → pass to XAI / END
    - 'Needs Revision' → back to analyser (if retries remain)
    - 'Rejected'       → END with error
    """
    validation = state.get("validation") or {}
    verdict    = validation.get("valid")
    retries    = state.get("retry_count", 0)

    if verdict == "Approved":
        logger.info("  ✅ Validator approved — proceeding")
        return "xai_node"

    if verdict == "Needs Revision" and retries < MAX_RETRIES:
        logger.warning(
            f"  ⚠️  Needs revision (attempt {retries + 1}/{MAX_RETRIES}) "
            f"— re-running analyser"
        )
        return "analyser_node"

    logger.error("  ❌ Rejected or max retries reached — ending pipeline")
    return END


# ── Graph assembly ────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(NDAState)

    graph.add_node("analyser_node",  analyser_node)
    graph.add_node("validator_node", validator_node)

    graph.add_edge(START,           "analyser_node")
    graph.add_edge("analyser_node", "validator_node")

    graph.add_conditional_edges(
        "validator_node",
        route_after_validator,
        {
            "analyser_node": "analyser_node",
            "xai_node"     : END,   # placeholder until XAI node is added
            END            : END,
        },
    )

    return graph