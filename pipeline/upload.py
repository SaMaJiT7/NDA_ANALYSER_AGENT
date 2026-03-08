from pipeline.embeddings import embed_query
import json
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType
)

load_dotenv()

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data")

COLLECTION_NAME  = "nda_policy_store"    # ← fixed: was COLLECTIONS_NAME
VECTOR_DIM       = 768                   # e5-base-v2 embedding dimension

# ── Singleton model ───────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        print("⏳ Loading e5-base-v2...")
        _model = SentenceTransformer("intfloat/e5-base-v2")
        print("✅ Model loaded")
    return _model


# ── Load embedded chunks ──────────────────────────
def load_embedded(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "Embedded_Chunks_final.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} embedded chunks from {path}")
    return data


# ── Connect to Qdrant ─────────────────────────────
def connect_qdrant():
    url     = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if url and api_key:
        client = QdrantClient(url=url, api_key=api_key)
        print("✅ Connected to Qdrant Cloud")
    else:
        client = QdrantClient(host="localhost", port=6333)
        print("✅ Connected to local Qdrant")
    return client


# ── Create collection + indexes ───────────────────
def create_collection(client):
    existing = [col.name for col in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        print(f"⚠️  Collection '{COLLECTION_NAME}' already exists. Deleting and re-creating.")
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config  = VectorParams(
            size     = VECTOR_DIM,
            distance = Distance.COSINE
        ),
    )

    # ── Create payload indexes ────────────────────
    # Required for keyword filtering on array fields
    # Without these — any filter returns 400 Bad Request
    client.create_payload_index(
        collection_name = COLLECTION_NAME,
        field_name      = "nda_clause_types",
        field_schema    = PayloadSchemaType.KEYWORD
    )
    client.create_payload_index(
        collection_name = COLLECTION_NAME,
        field_name      = "employee_risk",
        field_schema    = PayloadSchemaType.KEYWORD
    )

    print(f"✅ Collection '{COLLECTION_NAME}' created successfully")
    print(f"✅ Payload indexes created for nda_clause_types and employee_risk")


# ── Upload chunks ─────────────────────────────────
def upload_chunks(client, chunks):
    points = []

    for idx, chunk in enumerate(chunks):
        point = PointStruct(
            id     = idx,
            vector = chunk["embedding"],
            payload = {
                # ── Core ──────────────────────────
                "chunk_id"          : chunk["chunk_id"],
                "section"           : chunk["section"],
                "title"             : chunk["title"],
                "body"              : chunk["body"],
                "is_split"          : chunk.get("is_split", False),
                "total_chunks"      : chunk.get("total_chunks", 1),

                # ── Legal classification ──────────
                "act"               : chunk.get("act", "Indian Contract Act 1872"),
                "jurisdiction"      : chunk.get("jurisdiction", "India"),
                "chapter"           : chunk.get("chapter", ""),
                "legal_concept"     : chunk.get("legal_concept", ""),
                "employee_risk"     : chunk.get("employee_risk", "LOW"),
                "typically_void"    : chunk.get("typically_void", False),
                "voidable"          : chunk.get("voidable", False),
                "enforcement_likely": chunk.get("enforcement_likely", True),

                # ── NDA specific ──────────────────
                "nda_clause_types"  : chunk.get("nda_clause_types", []),
                "trigger_keywords"  : chunk.get("trigger_keywords", []),
                "agent_hint"        : chunk.get("agent_hint", ""),
            }
        )
        points.append(point)

    client.upsert(
        collection_name = COLLECTION_NAME,
        points          = points
    )

    print(f"✅ Uploaded {len(points)} points to '{COLLECTION_NAME}'")


# ── Verify upload + indexes ───────────────────────
def verify_upload(client):
    info = client.get_collection(COLLECTION_NAME)

    print(f"\n── Collection Info ──────────────────────────")
    print(f"  Name        : {COLLECTION_NAME}")
    print(f"  Points      : {info.points_count}")
    print(f"  Vector size : {info.config.params.vectors.size}")
    print(f"  Distance    : {info.config.params.vectors.distance}")

    # ── Verify indexes exist ──────────────────────
    print(f"\n── Payload Indexes ──────────────────────────")
    payload_schema = info.payload_schema

    if payload_schema:
        for field, schema in payload_schema.items():
            print(f"  ✅ {field} — {schema.data_type}")
    else:
        print(f"  ❌ No indexes found — filtering will fail")


# ── Test query ────────────────────────────────────
def test_query(client):
    print(f"\n── Query Test ───────────────────────────────")

    model       = get_model()    # ← singleton — no reload
    test_clause = (
        "You have a 2 year bond period and a penalty of "
        "6 months salary if you leave early."
    )
    query_vec = embed_query(test_clause, model).tolist()

    results = client.query_points(
        collection_name = COLLECTION_NAME,
        query           = query_vec,
        limit           = 3
    )

    print(f"  Query: \"{test_clause}\"")
    for rank, hit in enumerate(results.points, 1):
        print(
            f"    {rank}. {hit.payload['chunk_id']:<12} | "
            f"S.{hit.payload['section']:<4} | "
            f"score: {hit.score:.4f} | "
            f"risk: {hit.payload['employee_risk']}"
        )


# ── Main upload pipeline ──────────────────────────
def upload_process():
    # Step 1 — Load embedded chunks
    embedded_chunks = load_embedded()

    # Step 2 — Connect
    client = connect_qdrant()

    # Step 3 — Create collection + indexes
    create_collection(client)

    # Step 4 — Upload
    upload_chunks(client, embedded_chunks)

    # Step 5 — Verify collection + indexes
    verify_upload(client)

    # Step 6 — Test query
    test_query(client)


if __name__ == "__main__":
    upload_process()
    print("\n✅ Upload process complete. Your vector DB is ready for retrieval!")