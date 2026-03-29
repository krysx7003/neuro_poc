#!/usr/bin/env python3
"""
NSA Batch Indexer + Strict Metadata Extractor (STRICT VERSION)
Procesuje pliki *.txt -> rygorystycznie filtruje metadane (brak pustych pól) -> Qdrant
"""
import argparse
import hashlib
import json
import os
import re
import sys
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

# ─────────────────────────── STRICT METADATA PARSER ────────────────
ALLOWED_KEYS = [
    'Tytuł', 'Data orzeczenia', 'Sąd', 'Sędziowie', 'Sentencja', 
    'Symbol z opisem', 'Data wpływu', 'Treść wyniku', 'Uzasadnienie', 
    'Skarżony organ', 'Powołane przepisy', 'Hasła tematyczne'
]

LIST_KEYS = [
    'Sędziowie', 'Skarżony organ', 'Powołane przepisy', 'Hasła tematyczne'
]

def parse_court_document(text: str, filename: str) -> Dict[str, Any]:
    """Ekstrahuje metadane i rygorystycznie sprawdza kompletność treści."""
    lines = text.split('\n')
    metadata = {}
    current_key = None
    current_value = []
    
    known_text_blocks = ['Sentencja', 'Uzasadnienie', 'Teza', 'Wskazówki']
    title_captured = False

    for line in lines:
        line_stripped = line.strip()
        
        if not line_stripped and current_key is None:
            continue

        if not title_captured and '|' not in line and line_stripped not in known_text_blocks:
            metadata['Tytuł'] = line_stripped
            title_captured = True
            continue

        if line_stripped in known_text_blocks:
            if current_key:
                metadata[current_key] = '\n'.join(current_value).strip().rstrip('|')
            current_key = line_stripped
            current_value = []
            continue

        if '|' in line and current_key not in known_text_blocks:
            if current_key:
                metadata[current_key] = '\n'.join(current_value).strip().rstrip('|')
            
            parts = line.split('|', 1)
            current_key = parts[0].strip()
            val = parts[1].strip()
            current_value = [val]
            
            if val.endswith('|'):
                metadata[current_key] = val[:-1].strip()
                current_key = None
                current_value = []
        else:
            if current_key:
                if line_stripped.endswith('|') and current_key not in known_text_blocks:
                    current_value.append(line_stripped[:-1].strip())
                    metadata[current_key] = '\n'.join(current_value).strip()
                    current_key = None
                    current_value = []
                else:
                    current_value.append(line_stripped)

    if current_key:
        metadata[current_key] = '\n'.join(current_value).strip().rstrip('|')

    # 1. Filtrowanie dozwolonych kluczy
    filtered_metadata = {k: v for k, v in metadata.items() if k in ALLOWED_KEYS}
    
    # 2. Konwersja na listy (tak jak w extract_nsa_metadata.py)
    for key in LIST_KEYS:
        if key in filtered_metadata:
            raw_text = filtered_metadata[key]
            filtered_metadata[key] = [item.strip() for item in raw_text.split('\n') if item.strip()]

    # 3. RYGORYSTYCZNA WERYFIKACJA (STRICT MODE)
    # Sprawdzamy, czy wszystkie 12 kluczy istnieje I czy mają treść (nie są puste)
    invalid_keys = []
    for key in ALLOWED_KEYS:
        val = filtered_metadata.get(key)
        # Sprawdza: brak klucza, puste stringi (po strip), puste listy
        if val is None:
            invalid_keys.append(f"{key} (brak)")
        elif isinstance(val, str) and not val.strip():
            invalid_keys.append(f"{key} (pusty)")
        elif isinstance(val, list) and not val:
            invalid_keys.append(f"{key} (pusta lista)")

    if invalid_keys:
        return {"_error": "invalid_metadata", "_missing": invalid_keys}

    # Mapowanie na schemat Qdrant
    title_full = filtered_metadata['Tytuł']
    sig_match = re.match(r"^(.+?)(?:\s+-\s+(.+))?$", title_full)
    signature = sig_match.group(1).strip() if sig_match else title_full
    doc_title = sig_match.group(2).strip() if sig_match and sig_match.group(2) else "Wyrok"

    year = int(filtered_metadata['Data orzeczenia'].split('-')[0]) if '-' in filtered_metadata['Data orzeczenia'] else 0

    doc = {
        "doc_type": "nsa_judgment",
        "source_collection": "NSA",
        "source_file": filename,
        "signature": signature,
        "title": doc_title,
        "court": filtered_metadata['Sąd'],
        "judges": filtered_metadata['Sędziowie'],
        "year": year,
        "date_issued": filtered_metadata['Data orzeczenia'],
        "ruling": filtered_metadata['Sentencja'],
        "reasoning": filtered_metadata['Uzasadnienie'],
        "related_acts": filtered_metadata['Powołane przepisy'],
        "keywords": filtered_metadata['Hasła tematyczne'],
        "symbol": filtered_metadata['Symbol z opisem'],
        "date_received": filtered_metadata['Data wpływu'],
        "outcome": filtered_metadata['Treść wyniku'],
        "accused_body": filtered_metadata['Skarżony organ'],
        "summary": metadata.get('Teza', '') 
    }
    
    doc["content_text"] = f"Sentencja:\n{doc['ruling']}\n\nUzasadnienie:\n{doc['reasoning']}"
    return doc

# (Dalsza część kodu: chunk_text, sig_to_uuid, build_embed_text, build_payload, index_nsa_batch pozostaje bez zmian strukturalnych)

# ─────────────────────────── CHUNKING ───────────────────────────
def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[Dict]:
    if len(text) <= max_chars:
        return [{"chunk_text": text, "chunk_index": 0, "chunk_total": 1}]
    paras = re.split(r"\n\n+", text)
    chunks = []
    current = ""
    for para in paras:
        if len(current) + len(para) <= max_chars:
            current += "\n\n" + para
        else:
            if current: chunks.append(current.strip())
            overlap_start = max(0, len(current) - overlap)
            current = current[overlap_start:] + "\n\n" + para
    if current: chunks.append(current.strip())
    return [{"chunk_text": c, "chunk_index": i, "chunk_total": len(chunks)} for i, c in enumerate(chunks)]

def sig_to_uuid(sig: str) -> str:
    return str(uuid.UUID(bytes=hashlib.md5(f"nsa:{sig}".encode()).digest()))

def build_embed_text(doc: Dict) -> str:
    header = f"{doc.get('signature', '')} {doc.get('title', '')} | {doc.get('court', '')} | {doc.get('year', '')}"
    content = doc.get("chunk_text", doc.get("content_text", ""))[:5000]
    return f"{header}\n\n{content}"

def build_payload(raw_doc: Dict, chunk_info: Dict) -> Dict[str, Any]:
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
        "keywords": raw_doc.get("keywords", []),
        "keywords_text": ", ".join(raw_doc.get("keywords", [])),
        "symbol": raw_doc.get("symbol", ""),
        "date_received": raw_doc.get("date_received", ""),
        "outcome": raw_doc.get("outcome", ""),
        "accused_body": raw_doc.get("accused_body", []),
        "status": "opublikowane",
        "related_eu_acts": [],
        "related_uodo_rulings": [],
        "related_court_rulings": [],
        **{k: chunk_info[k] for k in ["chunk_index", "chunk_total"]},
    }

def index_nsa_batch(folder_path: str, qdrant_url: str, rebuild: bool = False, device: str = None):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🤖 Loading {EMBED_MODEL} on {device}")
    model = SentenceTransformer(EMBED_MODEL, device=device, trust_remote_code=True)
    dim = model.get_sentence_embedding_dimension()
    client = QdrantClient(url=qdrant_url, timeout=60)
    
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        print(f"📦 Creating {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

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
            if not next_off: break
            offset = next_off
        print(f"🔄 Already indexed: {len(done_sigs)} NSA docs")
    
    txt_files = list(folder.glob("*.txt"))
    print(f"📁 Found {len(txt_files)} .txt files")
    
    all_chunks = []
    skipped = 0

    for txt_file in sorted(txt_files):
        try:
            text = txt_file.read_text(encoding="utf-8")
            raw_doc = parse_court_document(text, txt_file.name)
            
            if "_error" in raw_doc:
                print(f"  [POMINIĘTO] {txt_file.name} -> Brakuje treści w: {', '.join(raw_doc['_missing'])}")
                skipped += 1
                continue

            chunks = chunk_text(raw_doc["content_text"])
            for chunk_info in chunks:
                doc_id = f"nsa:{raw_doc['signature']}:{chunk_info['chunk_index']}"
                if doc_id not in done_sigs:
                    chunk_info.update(raw_doc)
                    all_chunks.append(chunk_info)
        except Exception as e:
            print(f"  [BŁĄD] Plik {txt_file.name}: {e}")
            skipped += 1
    
    if not all_chunks:
        print(f"✅ Brak nowych dokumentów do indeksowania (Pominięto {skipped}).")
        return

    print(f"📝 Indeksowanie {len(all_chunks)} chunków...")
    texts = [build_embed_text(c) for c in all_chunks]
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE).tolist()
    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_vectors = vectors[i:i + BATCH_SIZE]
        points = []
        for chunk, vec in zip(batch_chunks, batch_vectors):
            raw_doc = {k: chunk[k] for k in chunk if k not in ["chunk_text", "chunk_index", "chunk_total"]}
            payload = build_payload(raw_doc, chunk)
            points.append(PointStruct(id=sig_to_uuid(payload["doc_id"]), vector=vec, payload=payload))
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ +{len(points)} ({(i+len(points))/len(all_chunks)*100:.0f}%)")
    
    print(f"\n🎉 Gotowe. Pominięto {skipped} plików.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch index NSA judgments")
    parser.add_argument("--folder", required=True, help="Path to NSA txt folder")
    parser.add_argument("--qdrant", default=QDRANT_URL)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    index_nsa_batch(args.folder, args.qdrant, args.rebuild, args.device)