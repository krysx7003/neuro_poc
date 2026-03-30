#!/usr/bin/env python3
"""
Batch converter: Processes ALL subfolders containing UODO 4-files (index.json, toc.json, search.json, 000_pl.xml)
and outputs uodo_scraper.py-compatible JSONL files.
"""

import json
import re
import xml.etree.ElementTree as ET
import os
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# ─────────────────────────── CONFIG ──────────────────────────────

# Same constants as previous script (RELATION_TO_GRAPH, _PUB_STATUS_MAP, _MONTHS)
RELATION_TO_GRAPH = {
    "quotes": "QUOTES", "quoted": "QUOTED_BY", "refers": "REFERS", "referred": "REFERRED_BY",
    "implements": "IMPLEMENTS", "implemented": "IMPLEMENTED_BY", "amends": "AMENDS", 
    "amended": "AMENDED_BY", "executes": "EXECUTES", "introduces": "INTRODUCES",
    "replaces": "REPLACES", "replaced": "REPLACED_BY",
}

_PUB_STATUS_MAP = {
    "final": "prawomocna", "nonfinal": "nieprawomocna",
    "published": "prawomocna", "archived": "prawomocna",
}

# ─────────────────────────── PARSING FUNCTIONS (unchanged) ──────────────────────────────

def multilang_str(field) -> str:
    if isinstance(field, dict):
        return field.get("pl") or field.get("en") or ""
    return str(field) if field else ""

def refid_to_signature(refid: str) -> str:
    m = re.search(r"uodo:(\d{4}):([\w_]+)$", refid)
    if not m:
        return refid
    year = m.group(1)
    code = m.group(2).upper().replace("_", ".")
    parts = [p for p in code.split(".") if p != year]
    if len(parts) >= 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.{year}"
    return f"{code}.{year}"

def parse_index_json(index_data: Dict) -> Dict:
    result = {
        "name": multilang_str(index_data.get("name", {})),
        "title_full": multilang_str(index_data.get("title", {})),
        "kind": index_data.get("kind", ""),
        "pub_workflow_status": index_data.get("publication", {}).get("status", ""),
        "entities": [], "keywords_list": [], "keywords": "", "refs": [],
        "term_decision_type": [], "term_violation_type": [], "term_legal_basis": [],
        "term_corrective_measure": [], "term_sector": [],
        "dates": index_data.get("dates", []),
    }
    
    for ent in index_data.get("entities", []):
        result["entities"].append({
            "title": multilang_str(ent.get("title", {})),
            "name": multilang_str(ent.get("name", {})),
            "function": ent.get("function", "other"),
        })
    
    kw_names = []
    for term in index_data.get("terms", []):
        name = multilang_str(term.get("name", {}))
        label = term.get("label", "")
        if name:
            kw_names.append(name)
        
        if label:
            prefix = label.split(".")[0]
            if prefix == "1": result["term_decision_type"].append(name)
            elif prefix == "2": result["term_violation_type"].append(name)
            elif prefix == "3": result["term_legal_basis"].append(name)
            elif prefix == "4": result["term_corrective_measure"].append(name)
            elif prefix == "9": result["term_sector"].append(name)
    
    result["keywords_list"] = kw_names
    result["keywords"] = ", ".join(kw_names)
    result["refs"] = index_data.get("refs", [])
    
    return result

def parse_search_json(search_data: Dict) -> Dict:
    dates = search_data.get("dates", {})
    return {
        "date_announcement": dates.get("announcement", ""),
        "date_publication": dates.get("publication", ""),
    }

def parse_xml_content(xml_path: Path) -> str:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        content = []
        for elem in root.iter():
            if elem.text: content.append(elem.text.strip())
            if elem.tail: content.append(elem.tail.strip())
        return "\n".join(line for line in content if line)
    except Exception:
        with open(xml_path, 'r', encoding='utf-8') as f:
            return f.read()

def parse_dates(dates_data: List[Dict]) -> Dict[str, str]:
    result = {"date_issued": "", "date_published": "", "date_effect": ""}
    for d in dates_data:
        use = d.get("use", "")
        val = d.get("date", "")
        if not val: continue
        if use == "announcement" and not result["date_issued"]:
            result["date_issued"] = val
        elif use == "publication" and not result["date_published"]:
            result["date_published"] = val
        elif use == "effect" and not result["date_effect"]:
            result["date_effect"] = val
    return result

def parse_refs(refs_data: List[Dict]) -> Dict:
    result = {"acts": [], "eu_acts": [], "court_rulings": [], "uodo_rulings": [], "edpb": [], "refs_full": []}
    
    for ref in refs_data:
        refid = ref.get("refid", "")
        relation = ref.get("relation", "refers")
        ref_type = ref.get("type", "direct")
        name = ref.get("name", "") or ""
        graph_rel = RELATION_TO_GRAPH.get(relation, "REFERS")
        entry = {"refid": refid, "relation": relation, "ref_type": ref_type, 
                "graph_relation": graph_rel, "name": name}
        
        if "urn:ndoc:pro:pl:durp:" in refid:
            m = re.search(r"durp:(\d{4}):(\d+)", refid)
            if m:
                sig = f"Dz.U. {m.group(1)} poz. {m.group(2)}"
                entry.update({"signature": sig, "category": "act"})
                if sig not in result["acts"]: result["acts"].append(sig)
                result["refs_full"].append(entry)
        elif "urn:ndoc:pro:eu:ojol:" in refid:
            m = re.search(r"ojol:(\d{4}):(\d+)", refid)
            if m:
                sig = f"EU {m.group(1)}/{m.group(2)}"
                entry.update({"signature": sig, "category": "eu_act"})
                if sig not in result["eu_acts"]: result["eu_acts"].append(sig)
                result["refs_full"].append(entry)
        elif "urn:ndoc:court:" in refid:
            sig = name or refid.split(":")[-1].replace("_", " ").upper()
            entry.update({"signature": sig, "category": "court_ruling"})
            if sig not in result["court_rulings"]: result["court_rulings"].append(sig)
            result["refs_full"].append(entry)
        elif "urn:ndoc:gov:pl:uodo:" in refid:
            sig = name or refid_to_signature(refid)
            entry.update({"signature": sig, "category": "uodo_ruling"})
            if sig not in result["uodo_rulings"]: result["uodo_rulings"].append(sig)
            result["refs_full"].append(entry)
        elif "urn:ndoc:gov:eu:edpb:" in refid:
            sig = name or refid.split(":")[-1]
            entry.update({"signature": sig, "category": "edpb"})
            if sig not in result["edpb"]: result["edpb"].append(sig)
            result["refs_full"].append(entry)
        else:
            entry.update({"signature": name or refid, "category": "other"})
            result["refs_full"].append(entry)
    
    return result

def extract_legal_status(keywords: str, pub_status: str) -> str:
    if pub_status in _PUB_STATUS_MAP:
        return _PUB_STATUS_MAP[pub_status]
    return "prawomocna" if "prawomocna" in keywords.lower() else "nieprawomocna"

# ─────────────────────────── SINGLE DECISION PROCESSOR ──────────────────────────────

def process_single_decision(folder_path: Path, output_file: Path) -> Dict:
    """Process one folder → one JSONL line (returns doc for stats)."""
    index_path = folder_path / "index.json"
    toc_path = folder_path / "toc.json"
    search_path = folder_path / "search.json"
    xml_path = folder_path / "000_pl.xml"
    
    # Validate files exist
    required_files = [index_path, toc_path, search_path, xml_path]
    if not all(f.exists() for f in required_files):
        raise FileNotFoundError(f"Missing files in {folder_path}: {[f.name for f in required_files if not f.exists()]}")
    
    # Load data
    with open(index_path, 'r', encoding='utf-8') as f: index_data = json.load(f)
    with open(search_path, 'r', encoding='utf-8') as f: search_data = json.load(f)
    with open(toc_path, 'r', encoding='utf-8') as f: toc_data = json.load(f)
    
    # Build doc (EXACT scraper format)
    doc_id = index_data.get("id", "")
    refid = index_data.get("refid", "")
    signature = refid_to_signature(refid)
    
    doc = {
        "doc_id": doc_id, "refid": refid, "signature": signature,
        "url": f"https://orzeczenia.uodo.gov.pl/document/{refid}/content",
        "source_collection": "UODO", "title": "", "title_full": "", "keywords": "",
        "keywords_list": [], "status": "", "pub_workflow_status": "", "kind": "",
        "date_issued": "", "date_published": "", "date_effect": "", "year": 0,
        "entities": [], "content_text": parse_xml_content(xml_path),
        "meta": index_data, "toc": toc_data,
        "refs_from_content": {"acts": [], "eu_acts": [], "court_rulings": [], "uodo_rulings": []},
        "refs_full": [], "related_legislation": [], "related_rulings": [],
        "term_decision_type": [], "term_violation_type": [], "term_legal_basis": [],
        "term_corrective_measure": [], "term_sector": [],
    }
    
    # Apply all parsing
    parsed_meta = parse_index_json(index_data)
    doc.update({
        "title": parsed_meta["name"], "title_full": parsed_meta["title_full"],
        "keywords": parsed_meta["keywords"], "keywords_list": parsed_meta["keywords_list"],
        "entities": parsed_meta["entities"], "kind": parsed_meta["kind"],
        "pub_workflow_status": parsed_meta["pub_workflow_status"],
        "term_decision_type": parsed_meta["term_decision_type"],
        "term_violation_type": parsed_meta["term_violation_type"],
        "term_legal_basis": parsed_meta["term_legal_basis"],
        "term_corrective_measure": parsed_meta["term_corrective_measure"],
        "term_sector": parsed_meta["term_sector"],
    })
    
    dates_parsed = parse_dates(index_data.get("dates", []))
    doc.update(dates_parsed)
    if doc["date_issued"]: doc["year"] = int(doc["date_issued"][:4])
    
    doc["status"] = extract_legal_status(doc["keywords"], doc["pub_workflow_status"])
    
    if parsed_meta["refs"]:
        refs_parsed = parse_refs(parsed_meta["refs"])
        doc["refs_from_content"].update({
            k: v for k, v in refs_parsed.items() 
            if k in ["acts", "eu_acts", "court_rulings", "uodo_rulings"]
        })
        doc["refs_full"] = refs_parsed["refs_full"]
        
        def _find_relation(refs_full, sig): 
            return next((r.get("relation", "refers") for r in refs_full if r.get("signature") == sig), "refers")
        
        doc["related_legislation"] = [
            {"type": "act", "signature": s, "relation": _find_relation(refs_parsed["refs_full"], s)} 
            for s in refs_parsed["acts"]
        ] + [
            {"type": "eu_act", "signature": s, "relation": _find_relation(refs_parsed["refs_full"], s)} 
            for s in refs_parsed["eu_acts"]
        ]
        doc["related_rulings"] = [
            {"type": "uodo_ruling", "signature": s, "relation": _find_relation(refs_parsed["refs_full"], s)} 
            for s in refs_parsed["uodo_rulings"]
        ] + [
            {"type": "court_ruling", "signature": s, "relation": _find_relation(refs_parsed["refs_full"], s)} 
            for s in refs_parsed["court_rulings"]
        ]
    
    # Append to output
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    return doc

# ─────────────────────────── BATCH PROCESSOR ──────────────────────────────

def batch_process(root_dir: Path, output_file: Path):
    """Scan root_dir/*/, process each valid subfolder."""
    root_dir = root_dir.resolve()
    
    # Clear output if exists
    if output_file.exists():
        output_file.unlink()
    
    processed = 0
    errors = 0
    
    print(f"🔍 Scanning {root_dir} for subfolders with 4 UODO files...")
    
    for subfolder in root_dir.iterdir():
        if not subfolder.is_dir():
            continue
            
        try:
            doc = process_single_decision(subfolder, output_file)
            processed += 1
            sig = doc["signature"]
            print(f"✅ {sig:25} ({len(doc['content_text']):6,} chars, {len(doc['refs_full'])} refs)")
            
        except Exception as e:
            errors += 1
            print(f"❌ {subfolder.name:25} ERROR: {e}")
    
    print(f"\n📊 SUMMARY: {processed} processed, {errors} errors → {output_file}")
    print(f"   Ready for uodo_scraper.py pipeline!")

# ─────────────────────────── CLI ──────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert UODO subfolders to JSONL")
    parser.add_argument("root_dir", help="Folder containing subfolders with 4 files each")
    parser.add_argument("-o", "--output", default="uodo_orzeczenia.jsonl", help="Output JSONL file")
    args = parser.parse_args()
    
    batch_process(Path(args.root_dir), Path(args.output))