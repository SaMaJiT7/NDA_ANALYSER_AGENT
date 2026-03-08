from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def load_metadata(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "section_metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Loaded metadata for {len(metadata)} sections")
    return metadata

def load_chunks(file_path=None):
    if file_path is None:
        file_path = os.path.join(DATA_DIR, "Chunks.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"✅ Loaded {len(chunks)} chunks from {file_path}")
    return chunks

def load_embeddings_model():
    print("Loading the e5-base-v2 model...")
    model = SentenceTransformer("intfloat/e5-base-v2")
    print("✅ Model loaded successfully!")
    return model


def attach_metadata(chunks, metadata):
    enriched = []

    for chunk in chunks:
        sec_num = chunk["section"]
        meta    = metadata.get(sec_num, {})
        enriched.append({
            **chunk,
            "act"               : "Indian Contract Act 1872",
            "jurisdiction"      : "India",
            "chapter"           : meta.get("chapter", ""),
            "chapter_title"     : meta.get("chapter_title", ""),
            "legal_concept"     : meta.get("legal_concept", ""),
            "employee_risk"     : meta.get("employee_risk", "LOW"),
            "typically_void"    : meta.get("typically_void", False),
            "voidable"          : meta.get("voidable", False),
            "enforcement_likely": meta.get("enforcement_likely", True),
            "nda_clause_types"  : meta.get("nda_clause_types", []),
            "trigger_keywords"  : meta.get("trigger_keywords", []),
            "agent_hint"        : meta.get("agent_hint", ""),
        })
    print(f"✅ Metadata attached to {len(enriched)} chunks")
    return enriched

def generate_embeddings(chunks, model):
    embedded = []

    for chunk in chunks:

        keywords_text = ", ".join(chunk.get("trigger_keywords", []))
        hint_text = chunk.get("agent_hint", "")
        clause_types  = ", ".join(chunk.get("nda_clause_types", []))

        text = (
            f"passage: "
            f"Section {chunk['section']} — {chunk['title']}. "
            f"{chunk['body']} "
            f"NDA clause types: {clause_types}. "    # ← adds NDA vocabulary
            f"Keywords: {keywords_text}. "           # ← adds trigger terms
            f"Note: {hint_text}"
        )

        # Truncate to safe char limit (~1900 chars ≈ 475 tokens)
        # Leaves buffer below 512 token hard limit
        if len(text) > 1900:
            text = text[:1900]

        embedding = model.encode(
            text,
            normalize_embeddings=True
        )

        embedded.append({
            **chunk,
            "embedding": embedding.tolist()   # numpy → list for JSON
        })

        print(f"  ✅ {chunk['chunk_id']:<12} | "
            f"{len(chunk['body']):>5} chars | "
            f"dim: {len(embedding)}")

    print(f"\n✅ Embedded {len(embedded)} chunks total")
    return embedded
    

def save_embeddings(embedded_chunks, path=None):

    """
    Saves full embedded chunks including vectors to JSON.
    This is your input for the vector DB upload step.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "Embedded_Chunks_final.json")

    with open(path,"w", encoding="utf-8") as f:
        json.dump(embedded_chunks,f, indent=2, ensure_ascii=False)
    
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"✅ Saved to {path} ({size_mb:.2f} MB)")


def embed_query(query, model):
    """
    Embeds a user query using the same model.
    This is used for similarity search against the chunk embeddings.
    """
    text = f"query: {query}"
    return model.encode(text, normalize_embeddings=True)

def sanity_check(embedded, model):
    print("\n── Sanity Check ─────────────────────────────")

    # Check dimensions
    sample = embedded[0]
    print(f"  chunk_id   : {sample['chunk_id']}")
    print(f"  section    : S.{sample['section']}")
    print(f"  dimensions : {len(sample['embedding'])}")
    print(f"  risk level : {sample['employee_risk']}")
    print(f"  first 5    : {sample['embedding'][:5]}")

    # Test retrieval with a real NDA clause
    test_clauses = [
        "Employee shall not join any competing firm for 24 months after termination",
        "Employee shall pay Rs 50 lakhs as penalty if confidential information is disclosed",
        "All disputes shall be resolved only through arbitration and employee waives right to approach courts",
    ]

    print("\n── Retrieval Test ───────────────────────────")

    for clause in test_clauses:
        query_vec = embed_query(clause, model)

        # Compute cosine similarity against all chunks
        scores = []
        for chunk in embedded:
            vec        = np.array(chunk["embedding"])
            similarity = float(np.dot(query_vec, vec))
            scores.append((chunk["chunk_id"], chunk["section"], similarity))

        # Top 3 results
        top3 = sorted(scores, key=lambda x: x[2], reverse=True)[:3]

        print(f"\n  Query: \"{clause[:60]}...\"")
        for rank, (chunk_id, sec, score) in enumerate(top3, 1):
            print(f"    {rank}. {chunk_id:<12} | S.{sec:<4} | score: {score:.4f}")


def embed_pipeline():
    chunks = load_chunks()
    metadata = load_metadata()
    enriched_chunks = attach_metadata(chunks, metadata)

    model = load_embeddings_model()

    embedded_chunks = generate_embeddings(enriched_chunks, model)
    save_embeddings(embedded_chunks)
    
    sanity_check(embedded_chunks, model)
    print("\n✅ embed.py complete — ready for vector DB upload")

if __name__ == "__main__":
    embed_pipeline()