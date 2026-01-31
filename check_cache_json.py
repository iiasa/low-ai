"""Quick check: is sector_to_comments_cache.json valid JSON / corrupted?"""
import json
import os
import sys

cache_file = os.path.join('paper4data', 'sector_to_comments_cache.json')
path = os.path.abspath(cache_file)

if not os.path.exists(path):
    print("File not found:", path)
    sys.exit(1)

size_mb = os.path.getsize(path) / (1024 * 1024)
print(f"File: {path}")
print(f"Size: {size_mb:.1f} MB")

# 1) First/last bytes
with open(path, 'rb') as f:
    head = f.read(500)
    f.seek(-500, 2)
    tail = f.read(500)

print("\nFirst 100 bytes (repr):", repr(head[:100]))
print("Last 100 bytes (repr):", repr(tail[-100:]))

if not head.lstrip().startswith(b'{'):
    print("\nWARNING: File does not start with '{' - may be corrupted or not JSON object.")
else:
    print("\nOK: File starts with '{'")

if not tail.rstrip().endswith(b'}'):
    print("WARNING: File does not end with '}' - may be truncated or corrupted.")
else:
    print("OK: File ends with '}'")

# 2) Try to parse
print("\nAttempting json.load()...")
try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("OK: JSON parsed successfully.")
    if isinstance(data, dict):
        print(f"  Top-level keys: {list(data.keys())[:15]}")
        for k in list(data.keys())[:3]:
            v = data[k]
            n = len(v) if isinstance(v, (list, dict)) else "N/A"
            print(f"  [{k!r}]: type={type(v).__name__}, len={n}")
    else:
        print(f"  Type: {type(data)}, len={len(data)}")
except json.JSONDecodeError as e:
    print(f"CORRUPTED / INVALID JSON: {e}")
    print(f"  Line {e.lineno}, column {e.colno}, msg: {e.msg}")
    if e.pos is not None:
        print(f"  Position in file: {e.pos}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
