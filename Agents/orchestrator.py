import os
import sys
import json
import logging
from datetime import datetime

# ── Fix Windows console encoding ─────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8') # type: ignore
    except AttributeError:
        pass
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END

from Agents.state import NDAState
from Agents.nodes import (
    segment_node,
    analyse_node,
    validate_node,
    explainable_node,
    respond_node,
)
from Agents.routers import (
    route_after_segment,
    route_after_analyse,
    route_after_validation,
    route_after_xai,
)


load_dotenv()

# ── Setup logging ─────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Retry node — increments counter before re-analysis ──
def retry_node(state: NDAState) -> NDAState:
    """Bumps retry_count and passes feedback back to analyser."""
    count = state.get("retry_count", 0) + 1
    logger.info(f"  🔄 Retry #{count}")
    return {**state, "retry_count": count}


# ── Build LangGraph ──────────────────────────────
def build_graph():
    """
    NDA Analyser Pipeline
    ─────────────────────
    START → segment → analyse → validate ──┐
                         ↑                  │
                         └── retry ←── (Needs Revision)
                                            │
                              (Approved) ──→ xai → respond → END
                              (Rejected) ──→ END
                              (Error)    ──→ respond → END
    """
    graph = StateGraph(NDAState)

    # ── Add nodes ─────────────────────────────────
    graph.add_node("segment",   segment_node)
    graph.add_node("analyse",   analyse_node)
    graph.add_node("validate",  validate_node)
    graph.add_node("retry",     retry_node)
    graph.add_node("xai",       explainable_node)
    graph.add_node("responder", respond_node)

    # ── Edges ─────────────────────────────────────
    # START → segment
    graph.add_edge(START, "segment")

    # segment → analyse | END
    graph.add_conditional_edges("segment", route_after_segment, {
        "analyse": "analyse",
        "stop"   : END,
    })

    # analyse → validate | END
    graph.add_conditional_edges("analyse", route_after_analyse, {
        "validate": "validate",
        "stop"    : END,
    })

    # validate → xai | retry | responder | END
    graph.add_conditional_edges("validate", route_after_validation, {
        "xai"      : "xai",
        "retry"    : "retry",
        "responder": "responder",
        "stop"     : END,
    })

    # retry → analyse (loop back)
    graph.add_edge("retry", "analyse")

    # xai → responder (always — errors handled inside)
    graph.add_conditional_edges("xai", route_after_xai, {
        "responder": "responder",
    })

    # responder → END
    graph.add_edge("responder", END)

    return graph.compile()


# ── Run pipeline ─────────────────────────────────
def run_pipeline(pdf_path: str) -> dict:
    """
    Entry point — takes a PDF path, returns final state.
    Uses stream() to capture each node's output individually.
    """
    logger.info(f"\n{'═' * 60}")
    logger.info(f"  NDA ANALYSER PIPELINE")
    logger.info(f"  Input: {pdf_path}")
    logger.info(f"{'═' * 60}\n")

    app = build_graph()

    initial_state: NDAState = {
        "pdf_path"           : pdf_path,
        "structured_nda"     : None,
        "risk_report"        : None,
        "validation"         : None,
        "xai_report"         : None,
        "final_response"     : None,
        "segment_output_path": None,
        "retry_count"        : 0,
        "error"              : None,
        "feedback"           : None,
        "failed_clauses"     : None,
    }

    # ── Stream to capture per-node outputs ────────
    node_outputs = {}   # {node_name: output_dict}
    final_state  = dict(initial_state)
    step_counter = 0

    for event in app.stream(initial_state):
        # event is {node_name: state_update_dict}
        for node_name, node_update in event.items():
            step_counter += 1
            logger.info(f"  📦 Step {step_counter} — captured output from '{node_name}'")

            # Store the raw update from this node
            # If a node runs multiple times (retry), append with suffix
            key = node_name
            if key in node_outputs:
                run_idx = 2
                while f"{node_name}_run{run_idx}" in node_outputs:
                    run_idx += 1
                key = f"{node_name}_run{run_idx}"

            node_outputs[key] = _serialisable_copy(node_update)

            # Merge into running state
            if isinstance(node_update, dict):
                final_state.update(node_update)

    # ── Save per-node outputs ─────────────────────
    _save_node_outputs(node_outputs, pdf_path)

    # ── Print final response ──────────────────────
    if final_state.get("final_response"):
        print(final_state["final_response"])
    elif final_state.get("error"):
        print(f"\n❌ Pipeline failed: {final_state['error']}")
    else:
        print("\n⚠️  Pipeline completed but no response generated")

    return final_state


def _serialisable_copy(obj):
    """Return a JSON-safe copy of obj, dropping non-serialisable values."""
    if isinstance(obj, dict):
        return {
            k: _serialisable_copy(v) for k, v in obj.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
    if isinstance(obj, list):
        return [_serialisable_copy(item) for item in obj]
    return obj


def _save_node_outputs(node_outputs: dict, pdf_path: str):
    """Save per-node outputs to data/node_outputs_<timestamp>.json"""
    try:
        base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path   = os.path.join(base_dir, "data", f"node_outputs_{ts}.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        payload = {
            "pdf_path"     : pdf_path,
            "timestamp"    : ts,
            "total_steps"  : len(node_outputs),
            "node_order"   : list(node_outputs.keys()),
            "node_outputs" : node_outputs,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✅ Node outputs saved to {out_path}")
    except Exception as e:
        logger.warning(f"  ⚠️  Could not save node outputs: {e}")


# ── CLI entry point ──────────────────────────────
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python -m Agents.orchestrator <path-to-nda.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    final_state = run_pipeline(pdf_path)

    # ── Save final state for debugging ────────────
    try:
        base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        state_path  = os.path.join(base_dir, "data", "pipeline_state.json")
        serialisable = {
            k: v for k, v in final_state.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Pipeline state saved to {state_path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not save pipeline state: {e}")
