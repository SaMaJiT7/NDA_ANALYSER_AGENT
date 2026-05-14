#!/usr/bin/env python3
import os
import sys

os.chdir(r"c:\NDA_Analyser")
sys.path.insert(0, r"c:\NDA_Analyser")

from dotenv import load_dotenv
load_dotenv()

print("TEST 1: Checking imports...")
try:
    from Agents.orchestrator import run_pipeline
    print("OK - Imports successful")
except Exception as e:
    print("ERROR - Import failed: " + str(e))
    sys.exit(1)

print("\nTEST 2: Environment variables...")
for k in ["GROQ_API_KEY", "GOOGLE_API_KEY"]:
    v = os.getenv(k)
    print("OK" if v else "ERROR" + " - " + k)

print("\nTEST 3: PDF files...")
pdfs = os.listdir(r"c:\NDA_Analyser\documents")
pdf_files = [p for p in pdfs if p.endswith('.pdf')]
print("OK - Found {} PDFs".format(len(pdf_files)))

print("\nTEST 4: Starting pipeline...")
pdf_path = r"c:\NDA_Analyser\documents\nda_example02.pdf"
print("Running: " + pdf_path)
print("(This may take 5-15 minutes)...\n")

try:
    final_state = run_pipeline(pdf_path)
    print("\n\nPIPELINE COMPLETE")
    print("=" * 50)
    
    # Check results
    results = {
        "structured_nda": "Segmentation",
        "risk_report": "Analysis",
        "validation": "Validation",
        "xai_report": "XAI",
        "final_response": "Response",
        "error": "Errors"
    }
    
    for key, name in results.items():
        if key == "error":
            if final_state.get(key):
                print("ERROR - {}: {}".format(name, final_state[key]))
        else:
            if final_state.get(key):
                print("OK - {}: Generated".format(name))
    
    print("\nFinal state keys: {}".format(list(final_state.keys())))
    
    # Check output files
    print("\nOutput files:")
    for f in os.listdir(r"c:\NDA_Analyser\data"):
        if f.endswith('.json'):
            fpath = os.path.join(r"c:\NDA_Analyser\data", f)
            size = os.path.getsize(fpath) / 1024
            print("  - {} ({:.1f} KB)".format(f, size))
    
except Exception as e:
    print("\nERROR during pipeline: " + str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
