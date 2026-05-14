#!/usr/bin/env python3
"""
Quick workflow test script
Tests each node of the pipeline sequentially
"""

import sys
import os
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all imports work"""
    logger.info("=" * 60)
    logger.info("TEST 1: Checking imports...")
    logger.info("=" * 60)
    try:
        from Agents.orchestrator import build_graph, run_pipeline
        from Agents.nodes import segment_node, analyse_node, validate_node, explainable_node, respond_node
        from Agents.state import NDAState
        logger.info("✅ All imports successful")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_pdf_exists():
    """Check if test PDF exists"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Checking test PDF...")
    logger.info("=" * 60)
    pdf_path = r"c:\NDA_Analyser\documents\nda_example02.pdf"
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        logger.info(f"✅ PDF found: {pdf_path} ({size:.2f} MB)")
        return pdf_path
    else:
        logger.error(f"❌ PDF not found: {pdf_path}")
        return None

def test_environment():
    """Check required environment variables"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Checking environment variables...")
    logger.info("=" * 60)
    
    required = {
        "GROQ_API_KEY": "Groq API",
        "GOOGLE_API_KEY": "Google Generative AI",
        "QDRANT_URL": "Qdrant vector DB",
        "QDRANT_API_KEY": "Qdrant auth",
    }
    
    all_ok = True
    for key, desc in required.items():
        if os.getenv(key):
            logger.info(f"✅ {key} ({desc}): Present")
        else:
            logger.warning(f"❌ {key} ({desc}): Missing")
            all_ok = False
    
    return all_ok

def test_pipeline_run(pdf_path):
    """Run the full pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Running full pipeline...")
    logger.info("=" * 60)
    
    try:
        from Agents.orchestrator import run_pipeline
        
        logger.info(f"Starting pipeline with: {pdf_path}")
        final_state = run_pipeline(pdf_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("=" * 60)
        
        # Check what was produced
        logger.info("\nFinal State Summary:")
        for key in ["structured_nda", "risk_report", "validation", "xai_report", "final_response", "error"]:
            value = final_state.get(key)
            if value:
                if isinstance(value, dict):
                    logger.info(f"  ✅ {key}: Generated ({len(str(value))} chars)")
                elif isinstance(value, str):
                    logger.info(f"  ✅ {key}: Generated ({len(value)} chars)")
                else:
                    logger.info(f"  ✅ {key}: Generated")
            else:
                logger.info(f"  ⚠️  {key}: Not generated")
        
        if final_state.get("error"):
            logger.error(f"\n❌ Pipeline Error: {final_state['error']}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}", exc_info=True)
        return False

def test_output_files():
    """Check if output files were created"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Checking output files...")
    logger.info("=" * 60)
    
    data_dir = r"c:\NDA_Analyser\data"
    expected_files = {
        "structured_nda.json": "Segmentation output",
        "risk_report.json": "Analysis output",
        "validation_result.json": "Validation output",
        "xai_report.json": "XAI output",
        "final_report.json": "Final combined report",
        "pipeline_state.json": "Pipeline state snapshot",
    }
    
    found = {}
    for filename, description in expected_files.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            logger.info(f"✅ {filename} ({description}): {size:.2f} KB")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                found[filename] = len(str(data))
            except:
                found[filename] = 0
        else:
            logger.warning(f"❌ {filename}: Not found")
    
    # Check for node_outputs_*.json
    import glob
    node_outputs = glob.glob(os.path.join(data_dir, "node_outputs_*.json"))
    if node_outputs:
        logger.info(f"✅ {len(node_outputs)} node_outputs file(s) found")
        for f in node_outputs:
            size = os.path.getsize(f) / 1024
            logger.info(f"   - {os.path.basename(f)}: {size:.2f} KB")
    else:
        logger.warning("❌ No node_outputs files found")
    
    return len(found) > 0

def main():
    """Run all tests"""
    logger.info("\n\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " NDA ANALYSER WORKFLOW TEST ".center(58) + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    # Test 1: Imports
    if not test_imports():
        logger.error("\n❌ Import test failed. Cannot proceed.")
        return False
    
    # Test 2: PDF
    pdf_path = test_pdf_exists()
    if not pdf_path:
        logger.error("\n❌ PDF test failed. Cannot proceed.")
        return False
    
    # Test 3: Environment
    if not test_environment():
        logger.warning("\n⚠️  Some environment variables are missing. Pipeline may fail.")
    
    # Test 4: Pipeline
    pipeline_ok = test_pipeline_run(pdf_path)
    
    # Test 5: Outputs
    files_ok = test_output_files()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("OVERALL RESULT")
    logger.info("=" * 60)
    
    if pipeline_ok and files_ok:
        logger.info("✅ WORKFLOW TEST PASSED - Pipeline is working correctly!")
        return True
    elif pipeline_ok:
        logger.warning("⚠️  WORKFLOW PARTIALLY SUCCESSFUL - Pipeline ran but some outputs missing")
        return True
    else:
        logger.error("❌ WORKFLOW TEST FAILED - Check errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
