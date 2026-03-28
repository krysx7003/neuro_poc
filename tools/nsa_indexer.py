#!/usr/bin/env python3
"""
NSA Batch Indexer — processes all *.txt files in folder → Qdrant uodo_decisions

Usage:
python nsa_batch_indexer.py --folder ./NSAreferencje/
python nsa_batch_indexer.py --folder ./NSAreferencje/ --rebuild
python nsa_batch_indexer.py --folder ./NSAreferencje/ --qdrant http://localhost:6333

Compatible with uodo_indexer.py schema. Adds doc_type="nsa_judgment"
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
    FieldCondition,
    Filter,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
import torch

# ─────────────────────────── CONFIG ──────────────────────────────
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "uodo_decisions"
EMBED_MODEL = "sdadas/mmlw-retrieval-roberta-large"
BATCH_SIZE = 32
CHUNK_MAX_CHARS = 4000
CHUNK_OVERLAP_CHARS = 300

# ─────────────────────────── PARSER ─────────────────────────────
def parse_nsa_judgment(text: str, filename: str) -> Dict[str, Any]:
    """Extracts structured data from NSA judgment text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    doc = {
        "doc_type": "nsa_judgment",
        "source_collection": "NSA",
        "source_file": filename,
        "signature": "",
        "title": "",
        "court": "",
        "judges": [],
        "year": 0,
        "date_issued": "",
        "content_text": "",
        "summary": "",  # Tezy
        "ruling": "",   # Sentencja
        "reasoning": "", # Uzasadnienie
        "related_acts": [],
        "keywords": [],
    }
    
    # 1. HEADER: "FPS 6/97 - Uchwała Składu Siedmiu Sędziów"
    header_match = re.match(r"([A-Z/ ]+\d+/\d{2}) - (.+)", lines[0])
    if header_match:
        doc["signature"] = header_match.group(1).strip()
        doc["title"] = header_match.group(2).strip()
    
    # 2. METADATA TABLE (Data orzeczenia|1997-09-22|)
    table_start = next((i for i, line in enumerate(lines) if "Data orzeczenia" in line), None)
    if table_start:
        for i in range(table_start, min(table_start + 15, len(lines))):
            line = lines[i]
            if "|" in line and len(line.split("|")) >= 3:
                parts = [p.strip() for p in line.split("|")]
                key, value = parts[1], parts[2]
                
                if key == "Data orzeczenia" and value:
                    doc["date_issued"] = value
                    doc["year"] = int(value.split("-")[0]) if "-" in value else 0
                elif key == "Sąd":
                    doc["court"] = value
                elif key == "Sędziowie":
                    doc["judges"] = [j.strip().rstrip("/").strip() for j in value.split("\n") if "/" in j or j.strip()]
    
    # 3. SECTIONS: Tezy|Sentencja|Uzasadnienie
    sections = {"tezy": "", "sentencja": "", "uzasadnienie": ""}
    current_section = None
    
    for line in lines:
        if line.startswith("Tezy"):
            current_section = "tezy"
        elif line.startswith("Sentencja"):
            current_section = "sentencja"
        elif line.startswith("Uzasadnienie"):
            current_section = "uzasadnienie"
        elif current_section and line:
            sections[current_section] += line + "\n"
    
    doc["summary"] = sections["tezy"].strip()
    doc["ruling"] = sections["sentencja"].strip()
    doc["reasoning"] = sections["uzasadnienie"].strip()
    
    # Full content
    doc["content_text"] = "\n\n### ".join([
        f"Tezy: {sections['tezy']}",
        f"Sentencja: {sections['sentencja']}",
        f"Uzasadnienie: {sections['uzasadnienie']}"
    ]).strip()
    
    # 4. REFERENCES (Powołane przepisy|...)
    refs_match = re.search(r"Powołane przepisy\|(.*?)(?=\n[A-Z]{3}|$)", text, re.DOTALL | re.IGNORECASE)
    if refs_match:
        acts_text = refs_match.group(1)
        doc["related_acts"] = [act.strip() for act in re.split(r"[.;]", acts_text) if act.strip()]
    
    return doc

# ─────────────────────────── CHUNKING ───────────────────────────
def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[Dict]:
    """Split long content into overlapping chunks."""
    if len(text) <= max_chars:
        return [{"chunk_text": text, "chunk_index": 0, "chunk_total": 1}]
    
    # Split by paragraphs/sections
    paras = re.split(r"\n\n+", text)
    chunks = []
    current = ""
    
    for para in paras:
        if len(current) + len(para) <= max_chars:
            current += "\n\n" + para
        else:
            if current:
                chunks.append(current.strip())
            overlap_start = max(0, len(current) - overlap)
            current = current[overlap_start:] + "\n\n" + para
    
    if current:
        chunks.append(current.strip())
    
    return [{"chunk_text": c, "chunk_index": i, "chunk_total": len(chunks)} for i, c in enumerate(chunks)]

# ─────────────────────────── QDRANT HELPERS ─────────────────────
def sig_to_uuid(sig: str) -> str:
    return str(uuid.UUID(bytes=hashlib.md5(f"nsa:{sig}".encode()).digest()))

def build_embed_text(doc: Dict) -> str:
    """Text for embedding: signature + metadata + content."""
    header = f"{doc.get('signature', '')} {doc.get('title', '')} | {doc.get('court', '')} | {doc.get('year', '')}"
    content = doc.get("chunk_text", doc.get("content_text", ""))[:5000]
    return f"{header}\n\n{content}"

def build_payload(raw_doc: Dict, chunk_info: Dict) -> Dict[str, Any]:
    """Qdrant payload compatible with uodo_indexer."""
    doc_id = f"nsa:{raw_doc.get('signature', Path(raw_doc.get('source_file', '')).stem)}"
    if chunk_info["chunk_index"] > 0:
        doc_id += f":chunk{chunk_info['chunk_index']}"
    
    return {
        "doc_type": "nsa_judgment",
        "doc_id": doc_id,
        "signature": raw_doc.get("signature", ""),
        "title": raw_doc.get("title", ""),
        "court": raw_doc.get("court", ""),
        "judges": raw_doc.get("judges", []),
        "year": raw_doc.get("year", 0),
        "date_issued": raw_doc.get("date_issued", ""),
        "content_text": chunk_info["chunk_text"],
        "summary": raw_doc.get("summary", ""),
        "ruling": raw_doc.get("ruling", ""),
        "source_file": raw_doc.get("source_file", ""),
        "related_acts": raw_doc.get("related_acts", []),
        "keywords": [],
        "keywords_text": "",
        # Compatible empty fields
        "status": "opublikowane",
        "related_eu_acts": [],
        "related_uodo_rulings": [],
        "related_court_rulings": [],
        **{k: chunk_info[k] for k in ["chunk_index", "chunk_total"]},
    }

# ─────────────────────────── MAIN ───────────────────────────────
def index_nsa_batch(folder_path: str, qdrant_url: str, rebuild: bool = False, device: str = None):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return
    
    # Load embedder
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🤖 Loading {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)
    dim = model.get_sentence_embedding_dimension()
    
    # Qdrant client
    client = QdrantClient(url=qdrant_url, timeout=60)
    
    # Create/update collection (same as uodo_indexer)
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        print(f"📦 Creating {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field, schema in [
            ("signature", PayloadSchemaType.KEYWORD),
            ("court", PayloadSchemaType.KEYWORD),
            ("year", PayloadSchemaType.INTEGER),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("keywords", PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(COLLECTION_NAME, field, schema)
    
    # Get already indexed NSA docs
    done_sigs = set()
    if not rebuild:
        offset = None
        while True:
            pts, next_off = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                scroll_filter=Filter(must=[FieldCondition(key="doc_type", match=MatchValue(value="nsa_judgment"))]),
                offset=offset,
                with_payload=["doc_id"],
            )
            for pt in pts or []:
                doc_id = pt.payload.get("doc_id", "")
                if doc_id.startswith("nsa:"):
                    done_sigs.add(doc_id)
            if not next_off:
                break
            offset = next_off
        print(f"🔄 Already indexed: {len(done_sigs)} NSA docs")
    
    # Process all .txt files
    txt_files = list(folder.glob("*.txt"))
    print(f"📁 Found {len(txt_files)} .txt files")
    
    all_chunks = []
    for txt_file in sorted(txt_files):
        print(f"📄 Processing {txt_file.name}...")
        text = txt_file.read_text(encoding="utf-8")
        raw_doc = parse_nsa_judgment(text, txt_file.name)
        
        if not raw_doc.get("signature"):
            print(f"  ⚠️ Skipped - no signature")
            continue
        
        chunks = chunk_text(raw_doc["content_text"])
        for chunk_info in chunks:
            doc_id = f"nsa:{raw_doc['signature']}:{chunk_info['chunk_index']}"
            if doc_id not in done_sigs:
                chunk_info.update(raw_doc)
                all_chunks.append(chunk_info)
    
    print(f"📝 {len(all_chunks)} new chunks to index")
    if not all_chunks:
        print("✅ All done!")
        return
    
    # Batch embedding + upsert
    indexed, errors = 0, 0
    texts = [build_embed_text(c) for c in all_chunks]
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE).tolist()
    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_vectors = vectors[i:i + BATCH_SIZE]
        points = []
        
        for chunk, vec in zip(batch_chunks, batch_vectors):
            raw_doc = {k: chunk[k] for k in chunk if k not in ["chunk_text", "chunk_index", "chunk_total"]}
            payload = build_payload(raw_doc, chunk)
            
            points.append(PointStruct(
                id=sig_to_uuid(payload["doc_id"]),
                vector=vec,
                payload=payload,
            ))
        
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            indexed += len(points)
            print(f"✅ +{len(points)} ({(i+len(points))/len(all_chunks)*100:.0f}%)")
        except Exception as e:
            print(f"❌ Batch error: {e}")
            errors += len(points)
    
    print(f"\n🎉 Indexed {indexed} chunks | Errors: {errors}")
    info = client.get_collection(COLLECTION_NAME)
    print(f"📊 Collection total: {info.points_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch index NSA judgments")
    parser.add_argument("--folder", required=True, help="Path to NSA txt folder")
    parser.add_argument("--qdrant", default=QDRANT_URL)
    parser.add_argument("--rebuild", action="store_true", help="Reindex everything")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    
    index_nsa_batch(args.folder, args.qdrant, args.rebuild, args.device)