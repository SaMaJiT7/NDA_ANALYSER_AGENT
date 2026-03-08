from typing import TypedDict, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import os
import logging
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.retriever import (
    retrieve_for_clause,
    retrieve_for_all_clauses,
    format_for_prompt
)

load_dotenv()

# ── Setup logging ─────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Model Loading ─────────────────────────────────
_model = None
def load_model():
    global _model
    if _model is None:
        llm = HuggingFaceEndpoint(
        repo_id=os.getenv("HF_MODEL", "deepseek-ai/DeepSeek-V3"),
        task="text-generation",
        max_new_tokens=1024,
        do_sample=False,
        repetition_penalty=1.03,
        ) # type: ignore
        logger.info("Loading model...")
        _model = ChatHuggingFace(llm=llm)
    return _model



def calculate_risk_score(assessed_clauses: list[dict]) -> dict:
    CLAUSE_RISK_WEIGHTS = {
    "non-compete": 1.45,       # highest — directly restricts livelihood
    "jurisdiction": 1.30,      # very high — removes legal recourse
    "indemnity": 1.20,         # high — potential for uncapped financial liability
    "penalty": 1.15,           # high — direct financial harm/liquidated damages
    "non-solicitation": 1.05,  # high — restricts career network
    "ip-ownership": 0.90,      # baseline — standard but critical for tech
    "termination": 0.80,       # baseline — governs exit and notice periods
    "confidentiality": 0.70,   # baseline — standard protection
    "data-protection": 0.60,   # compliance — regulatory requirements
    "definitions": 0.40,       # interpretive — low direct risk
    "general": 0.25,           # boilerplate — standard legal clauses
    "preamble": 0.10           # introductory — lowest risk
    }

    BASE_RISK  = {"HIGH": 30, "MEDIUM": 15, "LOW": 5}
    VOID_PENALTY = 10

    raw_score    = 0.0
    clause_scores = []
    high_count   = 0
    medium_count = 0
    low_count    = 0
    void_clauses = []

    for clause in assessed_clauses:
        assessment = clause.get("assessment")
        if not assessment or clause.get("skipped"):
            continue


        clause_type = clause.get("clause_type", "general")
        risk_level = assessment.get("risk_level", "LOW")
        retrieved   = clause.get("retrieved_sections", [])

        base_risk = BASE_RISK.get(risk_level, 5)
        
        type_weight = CLAUSE_RISK_WEIGHTS.get(clause_type, 0.5)

        # Component 3 — legal status multiplier
        legal_mult  = 1.0
        if assessment.get("is_void"):
            legal_mult += 0.3
        if assessment.get("is_voidable"):
            legal_mult += 0.2
        if not assessment.get("enforcement_likely"):
            legal_mult -= 0.1


        
        if not retrieved:
            con_weight = 1.2
        else:
            high_score = retrieved[0].get("score", 0.75)
            if high_score >= 0.85:
                con_weight = 1.0
            elif high_score >= 0.75:
                con_weight = 1.1
            else:
                con_weight = 1.2
            
        clause_score = (base_risk * type_weight * legal_mult * con_weight)

        if assessment.get("is_void"):
            clause_score += VOID_PENALTY
            void_clauses.append(clause.get("clause_title", "?"))

        
        raw_score += clause_score
        clause_scores.append({
            "clause_title" : clause.get("clause_title"),
            "clause_type"  : clause_type,
            "risk_level"   : risk_level,
            "clause_score" : round(clause_score, 2),
            "components"   : {
                "base"          : base_risk,
                "type_weight"   : type_weight,
                "legal_mult"    : round(legal_mult, 2),
                "conf_weight"   : con_weight,
                "void_penalty"  : VOID_PENALTY
                                  if assessment.get("is_void") else 0
            }
        })

        if risk_level == "HIGH":
            high_count += 1
        elif risk_level == "MEDIUM":
            medium_count += 1
        else:
            low_count += 1

    NORMALIZE_MAX = 200.0
    final_score   = min(round((raw_score / NORMALIZE_MAX) * 100), 100)

    if final_score >= 70:
        label          = "HIGH RISK"
        recommendation = "DO NOT SIGN without legal review"
    elif final_score >= 40:
        label          = "MEDIUM RISK"
        recommendation = "Negotiate flagged clauses before signing"
    else:
        label          = "LOW RISK"
        recommendation = "Standard NDA — review recommended"

    return {
            "risk_score"     : final_score,
            "raw_score"      : round(raw_score, 2),
            "risk_label"     : label,
            "recommendation" : recommendation,
            "clause_scores"  : clause_scores,   # ← per clause breakdown
            "breakdown"      : {
                    "high_clauses"  : high_count,
                    "medium_clauses": medium_count,
                    "low_clauses"   : low_count,
                    "void_clauses"  : void_clauses
        }
    }

class ClauseAssessment(BaseModel):
    risk_level         : str        = Field(description="HIGH | MEDIUM | LOW")
    is_void            : bool       = Field(description="True if clause is void under ICA")
    is_voidable        : bool       = Field(description="True if clause is voidable under ICA")
    enforcement_likely : bool       = Field(description="True if employer can realistically enforce")
    violated_sections  : List[str]  = Field(description="ICA sections violated e.g. ['S.27', 'S.28']")
    reasoning          : str        = Field(description="Detailed legal reasoning citing ICA sections")
    employee_impact    : str        = Field(description="Plain language explanation for employee")
    red_flags          : List[str]  = Field(description="Concerning phrases from the clause text")
    negotiation_points : List[str]  = Field(description="What employee should ask to change")


ANALYSER_TEMPLATE = """
You are a legal analyst specialising in Indian employment law and the Indian Contract Act 1872.

You will be given one NDA clause and relevant sections from the Indian Contract Act 1872.

Your job is to assess whether this clause is problematic for the EMPLOYEE.

IMPORTANT RULES:
- Always reason from the perspective of protecting the employee
- Cite specific ICA sections by number when making assessments
- Be direct — if a clause is VOID say it is VOID, do not hedge
- Base all reasoning on the retrieved ICA sections provided
- Do not invent legal precedents — only use what is provided

ADDITIONAL RULES:
- Assess the CONTENT of the clause, not just its heading
- If a termination clause contains confidentiality obligations— treat those obligations as confidentiality type for legal analysis
- If only part of a clause is void — set is_void: false and explain which specific sub-clause or phrase is void in reasoning
- Cite ALL relevant ICA sections — not just the primary one
- If clause survives termination indefinitely — always check S.27

RISK LEVELS:
- HIGH   : Clause is void, voidable, or significantly harmful to employee
- MEDIUM : Clause is enforceable but unfair or overly broad
- LOW    : Clause is standard and acceptable

NDA CLAUSE:
Title      : {clause_title}
Type       : {clause_type}
Clause Text: {clause_text}

RELEVANT ICA SECTIONS:
{retrieved_sections}

{format_instructions}"""

def analyser_chain():
    """
    Builds LangChain chain:
    PromptTemplate → deepseek-ai/DeepSeek-V3 → JsonOutputParser
    """
    model = _model or load_model()

    parser = JsonOutputParser(pydantic_object=ClauseAssessment)

    prompt = PromptTemplate.from_template(ANALYSER_TEMPLATE,
        partial_variables= {
            "format_instructions": parser.get_format_instructions()
        })

    chain = prompt | model | parser
    
    return chain

chain = None
def get_chain():
    global chain
    if chain is None:
        chain = analyser_chain()
    return chain

_gemini_model = None
def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_tokens=250)  # type: ignore
    return _gemini_model




def assess_clause(clause: dict) -> dict:

    clause_number = clause.get("clause_number", "?")
    clause_title  = clause.get("clause_title", "Untitled")
    clause_type   = clause.get("clause_type", "general")

    logger.info(
        f"  ⚖️  Assessing clause {clause_number} "
        f"— {clause_title} [{clause_type}]"
    )

    # Skip non-legal clauses
    if clause_type in {"preamble", "signature", "header"}:
        logger.info(f"     ↳ Skipped — {clause_type}")
        return {**clause, "retrieved_sections": [], "assessment": None, "skipped": True}

    # Step 1 — Retrieve
    retrieved = retrieve_for_clause(clause, top_k=3)

    if not retrieved:
        logger.warning(f"     ↳ ⚠️  No ICA sections retrieved")
        return {**clause, "retrieved_sections": [], "assessment": None, "skipped": True}

    # Step 2 — Format retrieved sections
    formatted_sections = format_for_prompt(retrieved, include_score=True)

    # Step 3 — Run LangChain chain
    try:
        chain      = get_chain()
        assessment = chain.invoke({
            "clause_title"      : clause_title,
            "clause_type"       : clause_type,
            "clause_text"       : clause.get("clause_text", ""),
            "retrieved_sections": formatted_sections
        })

        # assessment is already a validated dict — no JSON parsing needed
        logger.info(
            f"     ↳ ✅ Risk: {assessment.get('risk_level')} | "
            f"Void: {assessment.get('is_void')} | "
            f"Sections: {assessment.get('violated_sections')}"
        )
        logger.info(f"     ↳ Reasoning     : {assessment.get('reasoning', '')[:120]}...")
        logger.info(f"     ↳ Red Flags     : {assessment.get('red_flags', [])}")
        logger.info(f"     ↳ Negotiation   : {assessment.get('negotiation_points', [])}")

        return {
            **clause,
            "retrieved_sections": retrieved,
            "assessment"        : assessment,
            "skipped"           : False
        }

    except Exception as e:
        logger.error(f"     ↳ ❌ Chain failed: {e}")
        return {
            **clause,
            "retrieved_sections": retrieved,
            "assessment"        : None,
            "skipped"           : True
        }



# ── Extract company name from NDA ─────────────────
def extract_company_name(structured_nda: dict) -> str:
    """
    Tries to extract company name from preamble clause.
    Validator will use this for web search.
    """
    for clause in structured_nda.get("clauses", []):
        if clause.get("clause_type") == "preamble":
            # Pass preamble to LLM to extract company name
            try:
                prompt = (
                    f"You are a legal entity extractor. "
                    f"Identify the Disclosing Party (the company sharing confidential information) from the following NDA preamble. "
                    f"Output ONLY the company name. No explanation, no quotes, no 'The company is...' statements.\n\n"
                    f"TEXT: {clause.get('clause_text', '')[:2000]}"
                )
                response = get_gemini_model().invoke(prompt)

                company_name = response.content
                if not isinstance(company_name, str):
                    return "Unknown Entity"
                company_name = company_name.strip()
                # 3. Fallback check (Safety for your paper's results)
                if not company_name or len(company_name) < 2:
                    return "Unknown Entity"
                return company_name
            except Exception as e:
                print(f"Extraction Error: {e}")
                return "Extraction Failed"

    return "Unknown Entity"

# ── Main analyser function ────────────────────────
def run_analyser(structured_nda: dict) -> dict:
    """
    Full analyser pipeline.
    Assesses every clause and computes overall risk score.

    Args:
        structured_nda: structured NDA dict from orchestrator

    Returns:
        complete risk report dict
    """
    logger.info("\n── Analyser Starting ────────────────────────")
    logger.info(
        f"  NDA    : {structured_nda.get('nda_title', 'Unknown')}\n"
        f"  Clauses: {structured_nda.get('total_clauses', 0)}"
    )

    clauses = structured_nda.get("clauses", [])

    if not clauses:
        logger.error("❌ No clauses to assess")
        return {}
    
    # Step 1 — Extract company name for validator
    company_name = extract_company_name(structured_nda)
    logger.info(f"  Company: {company_name}")

    # Step 2 — Assess each clause
    assessed_clauses = []


    for clause in clauses:
        assessed = assess_clause(clause)

        assessed["clause_label"] = clause.get("clause_title", "") 
        
        assessed_clauses.append(assessed)

    # Step 3 — Compute overall risk score
    risk_summary = calculate_risk_score(assessed_clauses)

    logger.info(
        f"\n── Risk Summary ─────────────────────────────\n"
        f"  Score          : {risk_summary['risk_score']}/100\n"
        f"  Label          : {risk_summary['risk_label']}\n"
        f"  Recommendation : {risk_summary['recommendation']}\n"
        f"  HIGH clauses   : {risk_summary['breakdown']['high_clauses']}\n"
        f"  MEDIUM clauses : {risk_summary['breakdown']['medium_clauses']}\n"
        f"  LOW clauses    : {risk_summary['breakdown']['low_clauses']}\n"
        f"  VOID clauses   : {risk_summary['breakdown']['void_clauses']}"
    )

    # Step 4 — Build final report
    report = {
        "nda_title"      : structured_nda.get("nda_title", "Unknown"),
        "company_name"   : company_name,       # ← validator uses this
        "segmented_by"   : structured_nda.get("segmented_by", "unknown"),
        "risk_score"     : risk_summary["risk_score"],
        "risk_label"     : risk_summary["risk_label"],
        "recommendation" : risk_summary["recommendation"],
        "breakdown"      : risk_summary["breakdown"],
        "clause_reports" : assessed_clauses,
        "total_assessed" : len([
            c for c in assessed_clauses
            if not c.get("skipped")
        ]),
        "total_skipped"  : len([
            c for c in assessed_clauses
            if c.get("skipped")
        ])
    }

    logger.info("\n✅ Analyser complete — ready for validator")
    return report

def save_report(result: dict, path: str = "risk_report.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Risk report saved to {path}")

# ── Test ──────────────────────────────────────────
if __name__ == "__main__":

    # Load structured NDA from orchestrator output
    structured_nda_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "structured_nda.json")

    if not os.path.exists(structured_nda_path):
        logger.error(
            f"❌ {structured_nda_path} not found — "
            f"run orchestrator first"
        )
        exit(1)

    with open(structured_nda_path, "r", encoding="utf-8") as f:
        structured_nda = json.load(f)

    # Unwrap if the JSON root is a list
    if isinstance(structured_nda, list):
        structured_nda = structured_nda[0]

    # Run analyser
    report = run_analyser(structured_nda)



    # Print clause by clause summary
    print("\n── Clause Assessment Summary ────────────────")
    print(
        f"  {'#':<4} {'Type':<20} {'Risk':<8} "
        f"{'Void':<6} {'Sections'}"
    )
    print(f"  {'─' * 65}")

    for clause in report["clause_reports"]:
        assessment = clause.get("assessment")
        if not assessment:
            continue

        print(
            f"\n  Clause {clause['clause_number']} — {clause['clause_title']}"
            f"\n  Type     : {clause['clause_type']}"
            f"\n  Risk     : {assessment['risk_level']}"
            f"\n  Void     : {'YES' if assessment['is_void'] else 'NO'}"
            f"\n  Sections : {assessment['violated_sections']}"
            f"\n  Reasoning: {assessment['reasoning']}"
            f"\n  Impact   : {assessment['employee_impact']}"
            f"\n  Red Flags: {assessment['red_flags']}"
            f"\n  Negotiate: {assessment['negotiation_points']}"
            f"\n  {'─' * 65}"
        )

    print(f"\n── Final Verdict ────────────────────────────")
    print(f"  Score          : {report['risk_score']}/100")
    print(f"  Label          : {report['risk_label']}")
    print(f"  Recommendation : {report['recommendation']}")