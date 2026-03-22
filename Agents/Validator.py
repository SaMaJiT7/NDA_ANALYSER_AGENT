from typing import TypedDict, List, Optional,Literal, cast
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, SecretStr
import json
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()

# ── Setup logging ─────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    valid : Literal["Approved", "Needs Revision", "Rejected"] = Field(
        description=(
            "Approval status from validator. "
            "Approved = pass to XAI. "
            "Needs Revision = send back to analyser with feedback. "
            "Rejected = hard failure, cannot be fixed."
        )
    )
    feedback: Optional[str] = Field(None, description="Feedback for the Analyser Agent what to optimize in analysis.")
    confidence_score : float = Field(description="Confidence score 0.0-1.0 for the analysis result")
    citation_valid : bool = Field(
        description=(
            "True if all cited ICA sections exist in the rulebook "
            "and are correctly formatted as S.27, S.74 etc"
        )
    )
    risk_level : Optional[Literal["Low", "Medium", "High"]] = Field(None, description="Risk level associated with the analysis result as assessed by the validator agent.")
    relevance_score  : float = Field(       
        description=(
            "Relevance score 0.0-1.0 — how well retrieved ICA sections "
            "match the NDA clauses being analysed"
        )
    )
    failed_clauses : Optional[list[int]] = Field(None,description="Clause numbers that failed and need re-analysis")
    validation_summary : str               = Field(description="Overall validation summary")

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

_chain = None
_model = None

VALIDATOR_PROMPT = load_prompt("prompts/ValidatorPrompt.md")

def load_model():
    global _chain,_model
    if _chain is not None:
        return _chain

    logger.info("⏳ Loading validator model (Groq/DeepSeek)...")

    # llm = ChatOpenAI(
    #     model=os.getenv("Validator_MODEL"), # type: ignore
    #     api_key=SecretStr(os.getenv("GROQ_API_KEY") or ""),
    #     base_url="https://api.groq.com/openai/v1",
    #     temperature=0.0,
    #     model_kwargs={"max_tokens": 1024},
    # )
    llm = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V3"),
        task="text-generation",
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.03,
        ) # type: ignore
    logger.info("Loading model...")
    _model = ChatHuggingFace(llm=llm)

    # ── Use JsonOutputParser instead ──────────────
    parser = JsonOutputParser(pydantic_object=ValidationResult)

    prompt = PromptTemplate(
        template         = VALIDATOR_PROMPT,
        input_variables  = ["risk_report", "company_name"],
        partial_variables= {
            "format_instructions": parser.get_format_instructions()
        }
    )

    _chain = prompt | _model | parser

    logger.info("✅ Validator model loaded")
    return _chain


def prepare_report_for_prompt(risk_report: dict) -> dict:
    """
    Strips large fields before sending to validator prompt.
    Keeps only what validator needs to check.
    """

    clean = {
        "nda_title"    : risk_report.get("nda_title"),
        "company_name" : risk_report.get("company_name"),
        "risk_score"   : risk_report.get("risk_score"),
        "risk_label"   : risk_report.get("risk_label"),
        "breakdown"    : risk_report.get("breakdown"),
        "clause_reports": []
    }


    for clause in risk_report.get("clause_reports", []):
        if clause.get("skipped"):
            continue

        assessment = clause.get("assessment")
        if not assessment:
            continue

        clean["clause_reports"].append({
            "clause_number"    : clause.get("clause_number"),
            "clause_title"     : clause.get("clause_title"),
            "clause_type"      : clause.get("clause_type"),
            "assessment": {
                "risk_level"        : assessment.get("risk_level"),
                "is_void"           : assessment.get("is_void"),
                "is_voidable"       : assessment.get("is_voidable"),
                "enforcement_likely": assessment.get("enforcement_likely"),
                "violated_sections" : assessment.get("violated_sections"),
                "reasoning"         : assessment.get("reasoning"),
                "red_flags"         : assessment.get("red_flags"),
                "negotiation_points": assessment.get("negotiation_points")
            }
        })
    
    return clean


def validate_report(risk_report: dict) -> dict:
    """
    Generates validation prompt from risk report
    and sends to model for validation.

    Args:
        risk_report: full risk report from analyser

    Returns:
        ValidationResult dict
    """
    logger.info("\n── Validator Starting ───────────────────────")

    # ── Prepare report for prompt ─────────────────
    prepared_report = prepare_report_for_prompt(risk_report)

    company_name = risk_report.get("company_name", "Unknown Company")
    report_json = json.dumps(prepared_report, indent=2)

    logger.info(
        f"  Company     : {company_name}\n"
        f"  Clauses     : {risk_report.get('total_assessed', 0)} assessed\n"
        f"  Score       : {risk_report.get('risk_score', 0)}/100"
    )



    # ── Load model and run chain ──────────────────
    chain = load_model()
    
    try:
        result: dict = chain.invoke({
            "risk_report" : report_json,
            "company_name": company_name
        })

        icon = "✅" if result["valid"] == "Approved" else "⚠️ " if result["valid"] == "Needs Revision" else "❌"

        logger.info(
            f"\n── Validation Result ────────────────────────\n"
            f"  {icon} Status          : {result['valid']}\n"
            f"  Citation Valid   : {result['citation_valid']}\n"
            f"  Confidence Score : {result['confidence_score']}\n"
            f"  Relevance Score  : {result['relevance_score']}\n"
            f"  Risk Level       : {result['risk_level']}\n"
            f"  Failed Clauses   : {result.get('failed_clauses')}"
        )

        if result.get("feedback"):
            logger.warning(f"\n  Feedback       : {result['feedback']}")

        if result.get("failed_clauses"):
            logger.warning(f"\n  Failed Clauses : {result['failed_clauses']}")

        logger.info(f"\n  Summary : {result['validation_summary']}")

        return result

    except Exception as e:
        logger.error(f"  ❌ Validator chain failed: {e}")
        return {
            "valid"             : "Rejected",
            "feedback"          : f"Validator error: {e}",
            "confidence_score"  : 0.0,
            "citation_valid"    : False,
            "risk_level"        : None,
            "relevance_score"   : 0.0,
            "failed_clauses"    : None,
            "validation_summary": "Validation failed due to an internal error."
        }
    
    
def save_validation(result: dict, path: str = "validation_result.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Validation result saved to {path}")


# ── Test ──────────────────────────────────────────
if __name__ == "__main__":

    from Agents.Analyser import run_analyser, save_report

    # ── Step 1 — Load structured NDA ─────────────
    structured_nda_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "structured_nda.json")

    if not os.path.exists(structured_nda_path):
        logger.error("❌ structured_nda.json not found — run orchestrator first")
        exit(1)

    with open(structured_nda_path, "r", encoding="utf-8") as f:
        structured_nda = json.load(f)

    if isinstance(structured_nda, list):
        structured_nda = structured_nda[0]

    # # ── Step 2 — Run analyser ─────────────────────
    # logger.info("── Running Analyser ─────────────────────────")
    # risk_report = run_analyser(structured_nda)

    # if not risk_report:
    #     logger.error("❌ Analyser returned empty report")
    #     exit(1)

    # save_report(risk_report)
    # logger.info("✅ Analyser complete")
    risk_report_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "risk_report.json")
    with open(risk_report_path, "r", encoding="utf-8") as f:
        risk_report = json.load(f)
    
    # ── Step 3 — Run validator directly ──────────
    logger.info("\n── Running Validator ────────────────────────")
    validation_result = validate_report(risk_report)

    if not validation_result:
        logger.error("❌ Validator returned empty result")
        exit(1)

    save_validation(validation_result, path=os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "validation_result.json"))

    # ── Step 4 — Print final summary ──────────────
    print(f"\n── Final Summary ────────────────────────────")
    print(f"  Analyser Score  : {risk_report.get('risk_score')}/100")
    print(f"  Analyser Label  : {risk_report.get('risk_label')}")
    print(f"  Validator Status: {validation_result.get('valid')}")
    print(f"  Confidence      : {validation_result.get('confidence_score')}")
    print(f"  Citation Valid  : {validation_result.get('citation_valid')}")
    print(f"  Summary         : {validation_result.get('validation_summary')}")

    # ── Step 5 — Act on validation result ─────────
    if validation_result.get("valid") == "Approved":
        logger.info("\n✅ Report approved — ready for XAI agent")

    elif validation_result.get("valid") == "Needs Revision":
        logger.warning(
            f"\n⚠️  Needs revision\n"
            f"  Failed clauses : {validation_result.get('failed_clauses')}\n"
            f"  Feedback       : {validation_result.get('feedback')}"
        )

    else:
        logger.error(
            f"\n❌ Rejected\n"
            f"  Reason : {validation_result.get('feedback')}"
        )