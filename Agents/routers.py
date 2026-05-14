import logging
from Agents.state import NDAState, MAX_RETRIES


logger = logging.getLogger(__name__)


def route_after_segment(state: NDAState) -> str:
    """
    It is the Router function to route after segmentation
    of the NDA to the Analyser Agent.
    """
    if state.get("error"):
        logger.error(f"Routing after segmentation: Error found - {state['error']}")
        return "stop"
    return "analyse"


def route_after_analyse(state: NDAState) -> str:
    """
    It is the Router Function to route the analysis results to Validator Agent or the Explainer Agent.
    """
    if state.get("error"):
        logger.error(f"Routing failed analysis: Error found - {state['error']}")
        return "stop"
    risk_report = state.get("risk_report")
    if risk_report is not None and risk_report.get("violations"):
        return "validate"
    return "validate"

def route_after_validation(state: NDAState) -> str:
    """
    Router Function to send back to analyser if there is any mistake or
    just sent to the explainer agent to finalize the response.
    
    IMPROVED: After max retries, forces XAI generation instead of skipping.
    This ensures users always get explanations, even if validation isn't perfect.
    """
    if state.get("error"):
        logger.warning("  ⚠️  Validator error — sending to responder")
        return "responder"

    validation  = state.get("validation", {})
    status      = validation.get("valid", "Rejected") if validation is not None else "Unknown"
    retry_count = state.get("retry_count", 0)

    if status == "Approved":
        logger.info("Approved — proceeding to XAI")
        return "xai"

    elif status == "Needs Revision":
        if retry_count < MAX_RETRIES:
            logger.warning(
                f"  ⚠️  Needs Revision — "
                f"retry {retry_count + 1}/{MAX_RETRIES}"
            )
            return "retry"
        else:
            logger.warning(
                f"  ⚠️  Max retries ({MAX_RETRIES}) reached — "
                f"forcing XAI generation with current report"
            )
            return "xai"  # ← CHANGED: Force XAI instead of responder

    elif status == "Rejected":
        logger.error("Rejected — stopping pipeline")
        return "stop"

    return "stop"

def route_after_xai(state: NDAState) -> str:
    if state.get("error"):
        logger.warning("  ⚠️  XAI failed — proceeding without XAI")
    return "responder"

