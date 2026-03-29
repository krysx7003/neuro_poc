#!/usr/bin/env python3
"""
Signature Matcher: Compares 'signature' fields between JSONL files (FIXED for PowerShell)
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set

# Force immediate output flushing
sys.stdout.reconfigure(line_buffering=True)

def load_signatures(jsonl_file: Path) -> Set[str]:
    """Load all unique signatures from JSONL file."""
    signatures = set()
    line_count = 0
    
    print(f"🔄 Loading {jsonl_file}...", flush=True)
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                if line_num % 10000 == 0:
                    print(f"   Processed {line_num:,} lines...", flush=True)
                
                try:
                    doc = json.loads(line.strip())
                    sig = doc.get("signature", "").strip()
                    if sig:
                        signatures.add(sig)
                except json.JSONDecodeError:
                    pass  # Skip bad lines silently
        
        print(f"✅ {jsonl_file.name}: {len(signatures):,} unique signatures from {line_count:,} lines", flush=True)
        return signatures
        
    except FileNotFoundError:
        print(f"❌ File not found: {jsonl_file}", flush=True)
        return set()
    except Exception as e:
        print(f"❌ Error reading {jsonl_file}: {e}", flush=True)
        return set()

def analyze_matches(file1: Path, file2: Path):
    """Detailed comparison between two JSONL files."""
    print(f"\n{'='*80}", flush=True)
    print(f"🔍 COMPARING: {file1.name} vs {file2.name}", flush=True)
    print(f"{'='*80}", flush=True)
    
    sigs1 = load_signatures(file1)
    sigs2 = load_signatures(file2)
    
    if not sigs1 or not sigs2:
        print("❌ One or both files empty/failed to load", flush=True)
        return
    
    total1, total2 = len(sigs1), len(sigs2)
    
    # Matches (intersection)
    matches = sigs1 & sigs2
    match_pct1 = len(matches) / total1 * 100 if total1 else 0
    match_pct2 = len(matches) / total2 * 100 if total2 else 0
    
    # Unique to each
    only1 = sigs1 - sigs2
    only2 = sigs2 - sigs1
    
    print(f"\n📊 TOTALS:", flush=True)
    print(f"   {file1.name:25}: {total1:,} signatures", flush=True)
    print(f"   {file2.name:25}: {total2:,} signatures", flush=True)
    print()
    
    print(f"✅ MATCHES: {len(matches):,} ({match_pct1:5.1f}% of {file1.name}, {match_pct2:5.1f}% of {file2.name})", flush=True)
    if matches:
        print(f"   First 10 matches:", flush=True)
        for sig in sorted(matches)[:10]:
            print(f"      {sig}", flush=True)
        if len(matches) > 10:
            print(f"      ... and {len(matches)-10:,} more", flush=True)
    
    print(f"\n➡️  UNIQUE in {file1.name}: {len(only1):,} ({len(only1)/total1*100:.1f}%)", flush=True)
    if only1:
        print(f"   First 5: {', '.join(sorted(only1)[:5])}", flush=True)
    
    print(f"➡️  UNIQUE in {file2.name}: {len(only2):,} ({len(only2)/total2*100:.1f}%)", flush=True)
    if only2:
        print(f"   First 5: {', '.join(sorted(only2)[:5])}", flush=True)
    
    print(f"\n📈 SUMMARY TABLE:", flush=True)
    print(f"{'':<25} | {'Count':>8} | {'%':>6}", flush=True)
    print(f"{'─'*26}┼{'─'*10}┼{'─'*8}", flush=True)
    print(f"{file1.name:<25} | {total1:>8,} | {'100%':>6}", flush=True)
    print(f"{file2.name:<25} | {total2:>8,} | {'100%':>6}", flush=True)
    print(f"{'Matches':<25} | {len(matches):>8,} | {match_pct1:>6.1f}%", flush=True)
    print(f"{'Only file1':<25} | {len(only1):>8,} | {len(only1)/total1*100:>6.1f}%", flush=True)
    print(f"{'Only file2':<25} | {len(only2):>8,} | {len(only2)/total2*100:>6.1f}%", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Compare signature fields between JSONL files")
    parser.add_argument("files", nargs='+', help="JSONL files to compare")
    args = parser.parse_args()
    
    files = [Path(f) for f in args.files]
    
    print(f"🚀 Starting comparison of {len(files)} files...", flush=True)
    print(f"Files: {', '.join(f.name for f in files)}", flush=True)
    print("-" * 80, flush=True)
    
    if len(files) == 2:
        analyze_matches(files[0], files[1])
    else:
        print("ℹ️  Use exactly 2 files for detailed comparison", flush=True)
        print("   Or add --matrix flag for multi-file analysis", flush=True)

if __name__ == "__main__":
    main()