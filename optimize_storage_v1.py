#!/usr/bin/env python3
"""Optimize storage: remove duplicate frame_features versions, keep only compressed"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

# Track what's removed
removed = []
saved_bytes = 0

# Remove uncompressed frame_features files (keep v10 for reference)
for f in DATA.glob('frame_features_v*.jsonl'):
    if f.name == 'frame_features_v10.jsonl':
        print(f"Keeping latest: {f.name}")
    else:
        size = f.stat().st_size
        f.unlink()
        removed.append(f.name)
        saved_bytes += size
        print(f"Removed: {f.name} ({size/1024/1024:.1f}MB)")

# Check for other large duplicates
for pattern in ['*.jsonl.gz', '*.csv.gz']:
    files = list(DATA.glob(pattern))
    # Already compressed, keep

print(f"\nTotal removed: {len(removed)} files")
print(f"Storage saved: {saved_bytes/1024/1024:.1f}MB")

# Report current data directory size
total_size = sum(f.stat().st_size for f in DATA.rglob('*') if f.is_file() and '.venv' not in str(f))
print(f"Data dir size now: {total_size/1024/1024:.1f}MB")
