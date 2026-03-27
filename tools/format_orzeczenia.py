#!/usr/bin/env python3
"""
format_orzeczenia.py + UODO API - FIXED VERSION
Converts local search.json → unified JSONL EXACTLY like uodo_scraper.py
NUMBERS → KEYWORDS RESOLVED via meta.json/terms[]
"""

import json
import re
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from requests.auth import HTTPBasicAuth
import argparse

# --- CONFIGURATION ---
ROOT_DIRECTORY = "C:/Users/wrobl/Desktop/LEXSEARCH/NEUROPOC/daneUODO/orzeczenia"
OUTPUT_FILE = "uodo_orzeczenia.jsonl"

API_BASE = "https://orzeczenia.uodo.gov.pl/api"
DEFAULT_DELAY = 0.3
MAX_RETRIES = 3
TIMEOUT = 30

# --- HTTP HELPERS (EXACT from scraper) ---
def make_session(user: str = None, password: str = None) -> requests.Session:
    s = requests.Session()
    if user and password:
        s.auth = HTTPBasicAuth(user, password)
    s.headers["Accept"] = "application/json"
    return s

def get(session: requests.Session, url: str, retries: int = MAX_RETRIES, accept: str = None) -> Optional[requests.Response]:
    headers = {"Accept": accept} if accept else {}
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=TIMEOUT, headers=headers)
            if r.status_code == 200: return r
            if r.status_code == 404: return None
            if r.status_code == 401: 
                print(f"❌ HTTP 401 — wymagana autoryzacja (--user / --password)")
                return None
            print(f"⚠️ HTTP {r.status_code} dla {url} (próba {attempt+1})")
        except Exception as e:
            print(f"⚠️ Błąd połączenia: {e} (próba {attempt+1})")
        if attempt < retries - 1: time.sleep(2)
    return None

# --- PARSERS (EXACT from scraper) ---
def multilang_str(field) -> str:
    if isinstance(field, dict): return field.get("pl") or field.get("en") or ""
    return str(field) if field else ""

def parse_meta(data: Dict) -> Dict:
    result = {
        "name": "", "title_full": "", "keywords": "", "keywords_list": [],
        "entities": [], "kind": "", "legal_status": "", "pub_workflow_status": "",
        "date_issued": "", "date_published": "", "refs": {},
        "term_decision_type": [], "term_violation_type": [], "term_legal_basis": [],
        "term_corrective_measure": [], "term_sector": []
    }
    if not data: return result
    
    result["name"] = multilang_str(data.get("name", {}))
    result["title_full"] = multilang_str(data.get("title", {}))
    result["legal_status"] = data.get("status", "")
    
    # Parse terms → keywords + taxonomy (THIS RESOLVES NUMBERS!)
    kw_names = []
    for term in (data.get("terms", []) or []):
        if not isinstance(term, dict): continue
        name = multilang_str(term.get("name", {}))
        label = term.get("label", "")
        if name: kw_names.append(name)
        if label:
            prefix = label.split(".")[0]
            if prefix == "1": result["term_decision_type"].append(name)
            elif prefix == "2": result["term_violation_type"].append(name)
            elif prefix == "3": result["term_legal_basis"].append(name)
            elif prefix == "4": result["term_corrective_measure"].append(name)
            elif prefix == "9": result["term_sector"].append(name)
    
    result["keywords_list"] = kw_names
    result["keywords"] = ", ".join(kw_names)
    
    # Entities
    for ent in (data.get("entities", []) or []):
        if isinstance(ent, dict):
            result["entities"].append({
                "title": multilang_str(ent.get("title", {})),
                "name": multilang_str(ent.get("name", {})),
                "function": ent.get("function", "other")
            })
    
    result["kind"] = data.get("kind", "")
    pub = data.get("publication", {})
    result["pub_workflow_status"] = pub.get("status", "") if isinstance(pub, dict) else ""
    
    # Parse dates from meta (calls parse_dates)
    dates_parsed = parse_dates(data.get("dates", []))
    result["date_issued"] = dates_parsed["date_issued"]
    result["date_published"] = dates_parsed["date_published"]
    
    return result

def parse_dates(data) -> Dict[str, str]:
    result = {"date_issued": "", "date_published": "", "date_effect": ""}
    if not data: return result
    
    items = data if isinstance(data, list) else data.get("dates", [])
    for d in (items if isinstance(items, list) else []):
        if not isinstance(d, dict): continue
        use = d.get("use", "")
        val = d.get("date", "")
        if not val: continue
        if use == "announcement" and not result["date_issued"]: result["date_issued"] = val
        elif use == "publication" and not result["date_published"]: result["date_published"] = val
        elif use == "effect" and not result["date_effect"]: result["date_effect"] = val
    return result

def refid_to_signature(refid: str) -> str:
    m = re.search(r"uodo:(\d{4}):([\w_]+)$", refid)
    if not m: return refid
    year = m.group(1)
    code = m.group(2).upper().replace("_", ".")
    parts = [p for p in code.split(".") if p != year]
    if len(parts) >= 3: return f"{parts[0]}.{parts[1]}.{parts[2]}.{year}"
    return f"{code}.{year}"

def extract_legal_status(keywords: str, pub_status: str) -> str:
    _PUB_STATUS_MAP = {"final": "prawomocna", "nonfinal": "nieprawomocna", "published": "prawomocna", "archived": "prawomocna"}
    if pub_status in _PUB_STATUS_MAP: return _PUB_STATUS_MAP[pub_status]
    if "prawomocna" in keywords.lower(): return "prawomocna"
    return "nieprawomocna"

# --- EXACT fetch_decision FROM SCRAPER (KEYWORD RESOLUTION!) ---
def fetch_decision(session: requests.Session, doc_id: str, doc_fields: Dict, delay: float = DEFAULT_DELAY) -> Dict:
    refid = doc_fields.get("refid", "")
    if not refid: return {"_error": "brak_refid", "doc_id": doc_id}
    
    sig = refid_to_signature(refid)
    doc = {
        "doc_id": doc_id, "refid": refid, "signature": sig,
        "url": f"https://orzeczenia.uodo.gov.pl/document/{refid}/content",
        "source_collection": "UODO", "title": "", "title_full": "",
        "keywords": "", "keywords_list": [], "status": "", "pub_workflow_status": "",
        "kind": "", "date_issued": "", "date_published": "", "date_effect": "",
        "year": 0, "entities": [], "content_text": "", "meta": {},
        "refs_from_content": {"acts": [], "eu_acts": [], "court_rulings": [], "uodo_rulings": []},
        "refs_full": [], "related_legislation": [], "related_rulings": [],
        "term_decision_type": [], "term_violation_type": [], "term_legal_basis": [],
        "term_corrective_measure": [], "term_sector": []
    }
    
    # 1. Year from signature
    year_m = re.search(r"\b(20\d{2})\b", sig)
    doc["year"] = int(year_m.group(1)) if year_m else 0
    
    # 2. Initial data from local search.json (like scraper index)
    kw_raw = doc_fields.get("keywords", "")
    doc["keywords"] = ", ".join(kw_raw) if isinstance(kw_raw, list) else str(kw_raw or "")
    doc["title"] = doc_fields.get("title_pl", "")
    
    # 3. CONTENT body.txt (EXACT scraper sequence)
    r = get(session, f"{API_BASE}/documents/public/items/{refid}:0/body.txt", accept="text/plain")
    time.sleep(delay)
    if r:
        doc["content_text"] = r.text
        print(f"  ✅ body: {len(doc['content_text'])} chars")
    else:
        r = get(session, f"{API_BASE}/documents/public/items/{refid}/body.txt", accept="text/plain")
        time.sleep(delay)
        if r: 
            doc["content_text"] = r.text
            print(f"  ✅ body (fallback): {len(doc['content_text'])} chars")
    
    # 4. META.JSON - THIS RESOLVES NUMBERS → KEYWORDS!
    r = get(session, f"{API_BASE}/documents/public/items/{refid}/meta.json")
    time.sleep(delay)
    if r:
        meta_raw = r.json()
        doc["meta"] = meta_raw
        parsed = parse_meta(meta_raw)
        
        # EXACT scraper title logic
        if not doc["title"] and parsed["name"]: doc["title"] = parsed["name"]
        doc["title_full"] = parsed["title_full"] or doc["title"]
        
        # 🔑 KEYWORD RESOLUTION HAPPENS HERE!
        if parsed["keywords"]:  # meta.json gives Polish names!
            doc["keywords"] = parsed["keywords"]
            doc["keywords_list"] = parsed["keywords_list"]
            print(f"  ✅ meta: {len(doc['keywords_list'])} keywords resolved!")
        else:  # Fallback to local numeric keywords
            doc["keywords_list"] = [k.strip() for k in doc["keywords"].split(",") if k.strip()]
        
        # Copy all parsed fields (EXACT scraper)
        doc["entities"] = parsed["entities"]
        doc["kind"] = parsed["kind"]
        doc["pub_workflow_status"] = parsed["pub_workflow_status"]
        if parsed["date_issued"]: doc["date_issued"] = parsed["date_issued"]
        if parsed["date_published"]: doc["date_published"] = parsed["date_published"]
        
        # Status (EXACT scraper)
        if parsed["legal_status"]:
            doc["status"] = parsed["legal_status"]
        else:
            doc["status"] = extract_legal_status(doc["keywords"], doc["pub_workflow_status"])
        
        # Taxonomy
        for key in ["term_decision_type", "term_violation_type", "term_legal_basis", 
                   "term_corrective_measure", "term_sector"]:
            doc[key] = parsed[key]
    
    # 5. DATES.JSON
    r = get(session, f"{API_BASE}/documents/public/items/{refid}/dates.json")
    time.sleep(delay)
    if r:
        dates = parse_dates(r.json())
        doc["date_issued"] = dates["date_issued"] or doc["date_issued"]
        doc["date_published"] = dates["date_published"] or doc["date_published"]
        doc["date_effect"] = dates["date_effect"]
        if doc["date_issued"]: doc["year"] = int(doc["date_issued"][:4])
    
    return doc

def extract_refs_from_text(content: str, own_sig: str) -> Dict:
    result = {"acts": [], "eu_acts": [], "uodo_rulings": [], "court_rulings": [], "refs_full": []}
    if not content: return result
    
    # Dz.U.
    for m in re.finditer(r"Dz\.U\.\s*(?:z\s*r\.\s*)?(\d{4})\s*(?:poz\.)?\s*(\d+)", content):
        sig = f"Dz.U. {m.group(1)} poz. {m.group(2)}"
        if sig not in result["acts"]:
            result["acts"].append(sig)
            result["refs_full"].append({
                "signature": sig, "category": "act", "relation": "quotes", 
                "graph_relation": "QUOTES", "ref_type": "direct"
            })
    
    # RODO
    if re.search(r"2016/679|RODO|GDPR", content):
        sig = "EU 2016/679"
        if sig not in result["eu_acts"]:
            result["eu_acts"].append(sig)
            result["refs_full"].append({
                "signature": sig, "category": "eu_act", "relation": "implements",
                "graph_relation": "IMPLEMENTS", "ref_type": "direct", "name": "RODO"
            })
    
    # Other UODO decisions
    for m in re.finditer(r"(DKN|ZSPU|ZSZS|ZKE)\.[\d\.]+\d{4}", content):
        sig = m.group(0)
        if sig != own_sig and sig not in result["uodo_rulings"]:
            result["uodo_rulings"].append(sig)
            result["refs_full"].append({
                "signature": sig, "category": "uodo_ruling", "relation": "refers",
                "graph_relation": "REFERS", "ref_type": "direct"
            })
    
    return result

# --- MAIN TRANSFORM (uses scraper logic) ---
def transform_json(input_data: Dict, session: requests.Session) -> Optional[Dict]:
    content = input_data.get("fts", {}).get("content", {}).get("pl", "")
    if not content: return None
    
    sig_match = re.search(r'^([A-ZŚŻŹĆĄĘŁŃÓ]+(?:\.[\d]+)+)', content)
    if not sig_match: return None
    
    signature = sig_match.group(1)
    refid = input_data.get("refid", f"urn:ndoc:gov:pl:uodo:{signature.split('.')[-1]}:{signature.lower().replace('.', '_')}")
    
    # Prepare doc_fields like scraper index response
    doc_fields = {
        "id": input_data.get("id", f"local_{signature}"),
        "refid": refid,
        "keywords": input_data.get("keywords", []),
        "title_pl": content.split('\n')[0].strip()[:200]
    }
    
    print(f"🔍 Processing {signature}...")
    doc = fetch_decision(session, doc_fields["id"], doc_fields)
    
    if doc.get("_error"):
        print(f"  ❌ {doc['_error']}")
        return None
    
    # Fallback: use local content if API failed
    if not doc["content_text"]:
        doc["content_text"] = content[:50000]
    
    # Add doctype for your system
    doc["doctype"] = "uododecision"
    
    # Add refs fallback
    if not doc["refs_full"]:
        refs = extract_refs_from_text(doc["content_text"], signature)
        doc["refs_from_content"] = {
            "acts": refs["acts"], "eu_acts": refs["eu_acts"],
            "uodo_rulings": refs["uodo_rulings"], "court_rulings": refs["court_rulings"]
        }
        doc["refs_full"] = refs["refs_full"]
    
    print(f"  ✅ {len(doc['keywords_list'])} keywords: {doc['keywords_list'][:3]}...")
    return doc

# --- MAIN ---
def main(user: str = None, password: str = None, delay: float = 0.3, limit: int = None):
    session = make_session(user, password)
    path_root = Path(ROOT_DIRECTORY)
    
    count_files, count_docs, skipped_sig, api_errors = 0, 0, 0, 0
    
    print(f"🔍 Searching for 'search.json' in {path_root.absolute()}...")
    print(f"🌐 UODO API enabled (delay={delay}s)")
    
    global DEFAULT_DELAY
    DEFAULT_DELAY = delay
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for file_path in path_root.rglob('search.json'):
            count_files += 1
            print(f"\n📄 Processing: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    data = json.load(infile)
                
                items = data if isinstance(data, list) else [data]
                
                for i, item in enumerate(items[:limit] if limit else items):
                    standardized = transform_json(item, session)
                    if standardized is None:
                        skipped_sig += 1
                        print(f"  ⚠️ Skipped: No valid signature")
                    else:
                        outfile.write(json.dumps(standardized, ensure_ascii=False) + "\n")
                        count_docs += 1
                        print(f"  ✅ Saved: {standardized['signature']} ({len(standardized['keywords_list'])} keywords)")
                        
                        if limit and i >= limit-1: break
                        
            except Exception as e:
                print(f"❌ Error: {e}")
                api_errors += 1
    
    print("\n" + "="*60)
    print(f"✅ FINISHED!")
    print(f"📁 Files found: {count_files}")
    print(f"📝 Valid docs: {count_docs}")
    print(f"⚠️  Skipped sigs: {skipped_sig}")
    print(f"❌ API errors: {api_errors}")
    print(f"💾 Output: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="format_orzeczenia.py + UODO API (exact scraper replica)")
    parser.add_argument("--user", default=None, help="UODO API login")
    parser.add_argument("--password", default=None, help="UODO API password") 
    parser.add_argument("--delay", type=float, default=0.3, help="API delay (s)")
    parser.add_argument("--limit", type=int, default=None, help="Limit docs per file for testing")
    args = parser.parse_args()
    
    main(args.user, args.password, args.delay, args.limit)