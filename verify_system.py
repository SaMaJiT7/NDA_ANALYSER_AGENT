#!/usr/bin/env python3
"""Quick system verification without running full pipeline"""
import os
import sys
import json

sys.path.insert(0, r"c:\NDA_Analyser")
os.chdir(r"c:\NDA_Analyser")

from dotenv import load_dotenv
load_dotenv()

print("\n" + "="*60)
print("NDA ANALYSER - SYSTEM VERIFICATION")
print("="*60 + "\n")

# Check 1: Environment
print("1. Environment Variables:")
keys_check = ["GROQ_API_KEY", "GOOGLE_API_KEY", "QDRANT_URL"]
all_env_ok = True
for key in keys_check:
    val = os.getenv(key)
    if val:
        masked = val[:10] + "..." if len(val) > 10 else val
        print("   [OK] {}: {}".format(key, masked))
    else:
        print("   [FAIL] {}: NOT FOUND".format(key))
        all_env_ok = False

# Check 2: PDFs
print("\n2. Test Documents:")
pdf_dir = r"c:\NDA_Analyser\documents"
if os.path.exists(pdf_dir):
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print("   [OK] Found {} PDFs".format(len(pdfs)))
    for pdf in pdfs[:3]:
        path = os.path.join(pdf_dir, pdf)
        size_mb = os.path.getsize(path) / (1024*1024)
        print("       - {} ({:.2f} MB)".format(pdf, size_mb))
else:
    print("   [FAIL] Documents directory not found")

# Check 3: Data directory
print("\n3. Output Directory:")
data_dir = r"c:\NDA_Analyser\data"
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    json_files = [f for f in files if f.endswith('.json')]
    print("   [OK] Data directory exists")
    print("   [OK] {} JSON files present".format(len(json_files)))
else:
    print("   [FAIL] Data directory not found")

# Check 4: Imports
print("\n4. Python Imports:")
try:
    from Agents.orchestrator import run_pipeline, build_graph
    print("   [OK] Orchestrator imports successful")
except Exception as e:
    print("   [FAIL] Orchestrator import: {}".format(str(e)))

try:
    from Agents.nodes import segment_node, analyse_node, validate_node, explainable_node, respond_node
    print("   [OK] Node imports successful")
except Exception as e:
    print("   [FAIL] Node imports: {}".format(str(e)))

try:
    from Agents.state import NDAState
    print("   [OK] State model imports successful")
except Exception as e:
    print("   [FAIL] State model: {}".format(str(e)))

# Check 5: Recent outputs
print("\n5. Recent Pipeline Outputs:")
if os.path.exists(data_dir):
    try:
        if os.path.exists(os.path.join(data_dir, "final_report.json")):
            with open(os.path.join(data_dir, "final_report.json"), 'r') as f:
                data = json.load(f)
                if 'risk_report' in data:
                    print("   [OK] final_report.json exists and valid")
                    risk_score = data['risk_report'].get('risk_score', 'N/A')
                    risk_label = data['risk_report'].get('risk_label', 'N/A')
                    print("       Last run: Risk Score={}, Label={}".format(risk_score, risk_label))
    except:
        pass

# Summary
print("\n" + "="*60)
if all_env_ok:
    print("STATUS: READY TO RUN")
    print("\nTo start the pipeline:")
    print('python -m Agents.orchestrator "c:\\NDA_Analyser\\documents\\nda_example02.pdf"')
else:
    print("STATUS: MISSING CONFIGURATION")
    print("Please ensure all API keys are in .env file")

print("="*60 + "\n")
