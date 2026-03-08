import os
import logging
from typing import Optional
from dotenv import load_dotenv
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer

load_dotenv()
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

# ── Setup ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace with your actual credentials used in the previous script
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "nda_policy_store"

SECTION_DIVIDER  = "\n\n" + "─" * 40 + "\n\n"

SKIP_CLAUSE_TYPES = {"preamble", "signature", "header"}

CLAUSE_TYPE_MAP = {
    "non-compete"      : "non-compete",
    "non-solicitation" : "non-solicitation",
    "confidentiality"  : "confidentiality",
    "penalty"          : "penalty clause",
    "jurisdiction"     : "jurisdiction clause",
    "ip-ownership"     : "ip-ownership",
    "data-protection"  : "data-protection",
    "indemnity"        : "indemnity",
}

# ── Singletons — load once, reuse across calls ────
_model  = None
_client = None


def get_model():
    global _model
    if _model is None:
        logger.info("⏳ Loading e5-base-v2...")
        _model = SentenceTransformer("intfloat/e5-base-v2")
        logger.info("✅ Model loaded")
    return _model


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url     = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY")
        )
        logger.info("✅ Connected to Qdrant")
    return _client


# ── Core retrieve function ────────────────────────
def retrieve_sections(
    clause_text : str,
    top_k       : int           = 3,
    risk_filter : Optional[str] = None,
    clause_type : Optional[str] = None
) -> list[dict]:
    """
    Embeds clause text and retrieves top_k relevant ICA sections.

    Args:
        clause_text : full NDA clause text — never summarized
        top_k       : number of results (default 3)
        risk_filter : pre-filter by employee_risk — HIGH | MEDIUM | LOW
        clause_type : pre-filter by nda_clause_type

    Returns:
        list of dicts — each is one retrieved ICA section with score
    """
    model  = get_model()
    client = get_client()

    # ── Embed clause ──────────────────────────────
    # e5 requires "query: " prefix for search queries
    query_vec = model.encode(
        f"query: {clause_text}",
        normalize_embeddings=True
    ).tolist()

    # ── Build payload filters ─────────────────────
    filters = []

    if risk_filter:
        filters.append(
            FieldCondition(
                key   = "employee_risk",
                match = MatchValue(value=risk_filter)
            )
        )

    if clause_type:
        filters.append(
            FieldCondition(
                key   = "nda_clause_types",
                match = MatchValue(value=clause_type)
            )
        )

    query_filter = Filter(must=filters) if filters else None

    # ── Query Qdrant ──────────────────────────────
    raw = client.query_points(
        collection_name = COLLECTION_NAME,
        query           = query_vec,
        query_filter    = query_filter,
        limit           = top_k,
        with_payload    = True
    )

    # ── Extract points ────────────────────────────
    points = raw.points

    if not points:
        logger.warning("⚠️  No results returned from Qdrant")
        return []

    # ── Format results ────────────────────────────
    retrieved = []

    for hit in points:
        if not hit.payload:
            logger.warning(f"⚠️  Skipping hit {hit.id} — payload is None")
            continue
        try:
            retrieved.append({
                "chunk_id"          : hit.payload["chunk_id"],
                "section"           : hit.payload["section"],
                "title"             : hit.payload["title"],
                "body"              : hit.payload["body"],
                "score"             : round(hit.score, 4),
                "employee_risk"     : hit.payload["employee_risk"],
                "typically_void"    : hit.payload["typically_void"],
                "voidable"          : hit.payload["voidable"],
                "enforcement_likely": hit.payload["enforcement_likely"],
                "nda_clause_types"  : hit.payload["nda_clause_types"],
                "agent_hint"        : hit.payload["agent_hint"],
                "legal_concept"     : hit.payload["legal_concept"],
            })
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"⚠️  Skipping hit — missing field: {e}")
            continue

    return retrieved


# ── Format retrieved sections for LLM prompt ─────
def format_for_prompt(
    retrieved_sections : list[dict],
    include_score      : bool          = True,
    max_body_length    : Optional[int] = None
) -> str:
    """
    Converts retrieved ICA sections into clean readable
    text block for inclusion in analyser agent prompt.

    Args:
        retrieved_sections : list of section dicts from retrieve_sections
        include_score      : whether to include relevance score
        max_body_length    : truncate body text — None means no limit

    Returns:
        formatted string ready for LLM prompt
    """
    if not retrieved_sections:
        return "No relevant ICA sections found."

    formatted = []

    for i, sec in enumerate(retrieved_sections, 1):
        try:
            body = sec.get("body", "N/A")
            if max_body_length and len(body) > max_body_length:
                body = body[:max_body_length] + "..."

            lines = [
                f"[{i}] Section {sec.get('section', 'N/A')} "
                f"— {sec.get('title', 'Untitled')}",
                f"Legal Concept    : {sec.get('legal_concept', 'N/A')}",
                f"Employee Risk    : {sec.get('employee_risk', 'N/A')}",
                f"Typically Void   : {sec.get('typically_void', 'N/A')}",
                f"Enforcement Likely: {sec.get('enforcement_likely', 'N/A')}",
                f"Agent Hint       : {sec.get('agent_hint', 'N/A')}",
            ]

            if include_score:
                lines.append(
                    f"Relevance Score  : {sec.get('score', 0):.4f}"
                )

            lines.append(f"\nSection Text:\n{body}")
            formatted.append("\n".join(lines))

        except Exception as e:
            logger.warning(f"Error formatting section {i}: {e}")
            continue

    return SECTION_DIVIDER.join(formatted)


# ── Retrieve for single clause ────────────────────
def retrieve_for_clause(
    clause : dict,
    top_k  : int = 3
) -> list[dict]:
    """
    Retrieves relevant ICA sections for a single clause.
    Wrapper around retrieve_sections with clause dict input.

    Args:
        clause : clause dict with clause_text and clause_type
        top_k  : number of sections to retrieve

    Returns:
        list of retrieved section dicts
    """
    clause_type = clause.get("clause_type", "").lower()

    # Skip preamble, signature blocks etc
    if clause_type in SKIP_CLAUSE_TYPES:
        return []

    # Only filter known specific types
    # "general" / "other" → pure vector search, no metadata filter
    filter_type = CLAUSE_TYPE_MAP.get(clause_type, None)

    try:
        return retrieve_sections(
            clause_text = clause.get("clause_text", ""),
            top_k       = top_k,
            clause_type = filter_type
        )
    except Exception as e:
        logger.error(f"Retrieval failed for clause: {e}")
        return []


# ── Log retrieval result ──────────────────────────
def _log_retrieval_result(
    clause    : dict,
    retrieved : list[dict]
) -> None:
    """Logs the retrieval result for a single clause."""
    clause_num   = clause.get("clause_number", "?")
    clause_title = clause.get("clause_title", "Untitled")[:30]

    if retrieved:
        top = retrieved[0]
        logger.info(
            f"  ✅ Clause {clause_num:<3} "
            f"'{clause_title:<30}' → "
            f"{len(retrieved)} sections "
            f"(top: S.{top.get('section', '?')} "
            f"score: {top.get('score', 0):.4f})"
        )
    else:
        logger.warning(
            f"  ⚠️  Clause {clause_num} — no sections retrieved"
        )


# ── Retrieve for all clauses ──────────────────────
def retrieve_for_all_clauses(
    structured_nda : dict,
    top_k          : int  = 3,
    verbose        : bool = True
) -> dict:
    """
    Runs retrieval for every clause in structured NDA.
    Enriches each clause with retrieved_sections field.

    Args:
        structured_nda : parsed NDA dict with clauses list
        top_k          : number of ICA sections per clause
        verbose        : whether to log progress

    Returns:
        structured_nda enriched with retrieved_sections per clause
    """
    clauses = structured_nda.get("clauses", [])

    if not clauses:
        logger.warning("No clauses found in structured NDA")
        return structured_nda

    if verbose:
        logger.info(
            f"\n── Retrieving ICA sections for "
            f"{len(clauses)} clauses ────"
        )

    stats = {"total": len(clauses), "skipped": 0, "success": 0, "empty": 0}

    for clause in clauses:
        retrieved                  = retrieve_for_clause(clause, top_k)
        clause["retrieved_sections"] = retrieved

        clause_type = clause.get("clause_type", "").lower()

        if clause_type in SKIP_CLAUSE_TYPES:
            stats["skipped"] += 1
        elif retrieved:
            stats["success"] += 1
        else:
            stats["empty"] += 1

        if verbose and clause_type not in SKIP_CLAUSE_TYPES:
            _log_retrieval_result(clause, retrieved)

    if verbose:
        logger.info(
            f"\n── Retrieval Summary ────────────────────\n"
            f"  Total clauses : {stats['total']}\n"
            f"  Successful    : {stats['success']}\n"
            f"  Empty results : {stats['empty']}\n"
            f"  Skipped       : {stats['skipped']}"
        )

    return structured_nda


# ── Batch retrieve with progress bar ─────────────
def retrieve_for_all_clauses_with_progress(
    structured_nda : dict,
    top_k          : int = 3
) -> dict:
    """
    Same as retrieve_for_all_clauses but with tqdm progress bar.
    Useful for large NDAs with many clauses.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        logger.warning("tqdm not installed — using standard method")
        return retrieve_for_all_clauses(structured_nda, top_k)

    clauses = structured_nda.get("clauses", [])

    for clause in tqdm(clauses, desc="Retrieving ICA sections"):
        clause["retrieved_sections"] = retrieve_for_clause(clause, top_k)

    return structured_nda



# ── Test ──────────────────────────────────────────
if __name__ == "__main__":

    TEST_CLAUSES = [
        {
            "clause_text" : "Employee shall not join any competing firm "
                            "for 24 months after termination",
            "clause_type" : "non-compete",
            "expected"    : "27"
        },
        {
            "clause_text" : "Employee shall pay Rs 50 lakhs as penalty "
                            "if confidential information is disclosed",
            "clause_type" : "penalty",
            "expected"    : "74"
        },
        {
            "clause_text" : "All disputes shall be resolved only through "
                            "arbitration and employee waives right to "
                            "approach any court or tribunal",
            "clause_type" : "jurisdiction",
            "expected"    : "28"
        },
        {
            "clause_text" : "Employee signed this agreement under pressure "
                            "from employer who threatened to withdraw job offer. "
                            "Employer used authority and influence to compel signing.",
            "clause_type" : "general",
            "expected"    : "16"
        },
    ]

    print("── Retrieve Tool Test ───────────────────────")
    all_passed = True

    for test in TEST_CLAUSES:
        results = retrieve_for_clause(
            clause = {
                "clause_text" : test["clause_text"],
                "clause_type" : test["clause_type"]
            },
            top_k = 3
        )

        top_section = results[0]["section"] if results else "none"
        passed      = top_section == test["expected"]
        status      = "✅" if passed else "❌"

        if not passed:
            all_passed = False

        print(f"\n  {status} Type     : {test['clause_type']}")
        print(f"     Query    : {test['clause_text'][:55]}...")
        print(f"     Expected : S.{test['expected']}")
        print(f"     Got      : S.{top_section} "
              f"score: {results[0]['score'] if results else 'N/A'}")
        print(f"     Top 3    :")

        for r in results:
            void_flag = "VOID" if r["typically_void"] else "valid"
            print(f"       S.{r['section']:<4} | "
                  f"{r['score']} | "
                  f"{r['employee_risk']:<6} | "
                  f"{void_flag:<5} | "
                  f"{r['title'][:30]}")

    print(f"\n── Summary ──────────────────────────────────")
    if all_passed:
        print("✅ All tests passed — retrieve.py ready")
    else:
        print("⚠️  Some tests failed — check embeddings")

    # ── Test format_for_prompt ────────────────────
    print(f"\n── Prompt Format Test ───────────────────────")
    sample = retrieve_sections(
        "Employee shall not disclose any confidential information",
        top_k = 2
    )
    print(format_for_prompt(sample))