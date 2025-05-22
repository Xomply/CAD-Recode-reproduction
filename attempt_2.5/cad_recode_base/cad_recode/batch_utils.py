"""
Batch Utilities – conceptual skeleton

Goal: Convert thousands of tiny *.py* CADQuery scripts into ~MB-sized *.jsonl*
archives to satisfy DelftBlue 512 KiB chunk constraint and reduce inode count.

CLI usage (example):
    python -m cad_recode.data.batch_utils \
        --src /path/to/original_dataset/train \
        --dst /path/to/batched_dataset/train_batches \
        --batch_size 1000           # number of scripts per JSONL file

Main functions
--------------
• batchify_split(src_split_dir: str, dst_dir: str, batch_size: int = 1000)
    – Walk *.py* files inside *src_split_dir* (including nested folders).
    – Group `batch_size` scripts into a Python list of dicts `{code: ..., name: ...}`.
    – Dump list to f"batch_{idx:05d}.jsonl" (one JSON object per line).

• unbatch_jsonl(jsonl_path: str) → List[str]
    – Utility to read a batched file and return list of code strings.

Script entry-point
------------------
if __name__ == "__main__":
    parse CLI args (src, dst, batch_size) and call batchify_split for train/val/test.

Notes
-----
• Validation/test splits are treated the same.
• Preserve *relative path* metadata in JSON object so original file name remains.
• Provide progress bars via tqdm.
"""

from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                             core functions                                  #
# --------------------------------------------------------------------------- #
def _iter_py_files(root: Path):
    """Yield all .py file paths recursively under *root* (sorted for stability)."""
    for p in sorted(root.rglob("*.py")):
        yield p

def batchify_split(src_split_dir: str | os.PathLike,
                   dst_dir: str | os.PathLike,
                   batch_size: int = 1000) -> None:
    """
    Convert individual *.py files under *src_split_dir* into JSONL archives.

    Output files will be named batch_00000.jsonl, batch_00001.jsonl, …
    Each line is a JSON object:  {"name": "<relative/path.py>", "code": "<content>"}
    """
    src_split_dir = Path(src_split_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    buffer = []
    file_idx = 0

    for py_path in tqdm(_iter_py_files(src_split_dir),
                        desc=f"Batchifying {src_split_dir.name}"):
        rel_name = py_path.relative_to(src_split_dir).as_posix()
        code_txt = py_path.read_text()

        buffer.append({"name": rel_name, "code": code_txt})

        if len(buffer) >= batch_size:
            _flush_buffer(buffer, dst_dir, file_idx)
            file_idx += 1
            buffer.clear()

    if buffer:  # leftovers
        _flush_buffer(buffer, dst_dir, file_idx)

def _flush_buffer(entries: list[dict], dst_dir: Path, idx: int):
    """Write current buffer to JSONL file `batch_<idx>.jsonl`."""
    out_path = dst_dir / f"batch_{idx:05d}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for obj in entries:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")

def unbatch_jsonl(jsonl_path: str | os.PathLike) -> List[str]:
    """Return list of CADQuery code strings from a JSONL archive."""
    codes: list[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            codes.append(obj["code"])
    return codes

# --------------------------------------------------------------------------- #
#                               CLI                                           #
# --------------------------------------------------------------------------- #
def _parse_args():
    ap = argparse.ArgumentParser(description="Batchify CADQuery scripts into JSONL.")
    ap.add_argument("--src", required=True, help="Path to original split dir (train/val/test)")
    ap.add_argument("--dst", required=True, help="Destination folder for JSONL batch files")
    ap.add_argument("--batch_size", type=int, default=1000, help="Scripts per JSONL")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    batchify_split(args.src, args.dst, args.batch_size)
    print("✅ Done")
