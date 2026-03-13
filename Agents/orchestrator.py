import os
import sys
import json
import logging
from typing import TypedDict, Optional

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