#!/usr/bin/env python

"""
Build a 72 × 5 × 132 tensor from one or more peer‑review CSVs.

Usage
-----
python surveytensor.py input1.csv input2.csv --out peer_tensor.npy
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────── Constants ──────────────────────────────── #

TASK_REGEX = re.compile(
    r"Review the following team submissions.*?assess it \[(?P<task>(?:[1-9]|1[0-9]|2[0-4])(?:[AB])?)\]$"
)
TENSOR_SHAPE = (72, 5, 133)
VALID_RESPONSES = {"1", "2", "3", "4", "5"}


# ───────────────────────────── Loader helpers ───────────────────────────── #


def load_frames(paths: List[str]) -> List[pd.DataFrame]:
    """Read CSVs verbatim (preserve duplicate headers)."""
    return [pd.read_csv(p, dtype=str, header=0) for p in paths]


# ───────────────────────── PID index utilities ─────────────────────────── #


def build_pid_index(frames: List[pd.DataFrame]) -> Tuple[List[str], Dict[str, int]]:
    """Return (ordered PID list, lookup dict)."""
    pid_list, seen = [], set()
    for df in frames:
        # assumes PID in col‑0
        for pid in df.iloc[:, 1].astype(str).str.strip():
            if pid and pid not in seen:
                seen.add(pid)
                pid_list.append(pid)
    print(f"PID list: {pid_list}")
    return pid_list, {p: k for k, p in enumerate(pid_list)}


# ───────────────────────── Task/column mapping ─────────────────────────── #


def map_task_columns(df: pd.DataFrame) -> Dict[int, int]:
    """
    Map column index → I‑index (0‒71).
    Handles both ‘1‑24’ and ‘1A‑24B’.
    """
    mapping = {}
    for col_idx, col_name in enumerate(df.columns):
        m = TASK_REGEX.search(col_name)
        if not m:
            continue
        task_id = m.group("task")
        print(f"current task: {task_id}")
        if task_id.isdigit():  # 1‑24
            i = int(task_id) - 1  # 0‑23
        else:  # 1A‑24B
            base = int(task_id[:-1])  # 1‑24
            suffix = task_id[-1].upper()  # A/B
            i = 24 + (base - 1) * 2 + (suffix == "B")
            print(f"index for {task_id}: {i}")
        mapping[col_idx] = i
    print(f"column mapping: {mapping}")
    return mapping


# ───────────────────────── Tensor construction ─────────────────────────── #


def build_tensor(
    frames: List[pd.DataFrame],
    pid_lookup: Dict[str, int],
    shape: Tuple[int, int, int] = TENSOR_SHAPE,
) -> np.ndarray:
    """Return a populated (72, 5, 132) tensor of uint8."""
    tensor = np.zeros(shape, dtype=np.uint8)

    for df in frames:
        task_cols = map_task_columns(df)
        pid_series = df.iloc[:, 1].astype(str).str.strip()

        for row_idx, pid in enumerate(pid_series):
            if pid not in pid_lookup:
                continue  # or warn
            k = pid_lookup[pid]

            for col_idx, cell in enumerate(df.iloc[row_idx, :]):
                if col_idx not in task_cols or cell not in VALID_RESPONSES:
                    continue
                i = task_cols[col_idx]
                j = int(cell) - 1  # 1→0 … 5→4
                tensor[i, j, k] = 1
                print(f"For PID: {pid}, Tensor index [{i}][{j}][{k}] set to 1")
    for k, task in enumerate(tensor[0]):
        print(f"Task #: {k}")
        print(task)
        print(len(task))
    print(len(tensor))
    return tensor



# ──────────────────────────────── CLI glue ─────────────────────────────── #


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build peer‑review tensor")
    ap.add_argument("csvs", type=Path, nargs="+", help="Input CSV paths")
    ap.add_argument(
        "--out",
        "-o",
        type=Path,
        default="peer_tensor.npy",
        help="Output .npy file (default: peer_tensor.npy)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_frames([str(p) for p in args.csvs])
    _, pid_lookup = build_pid_index(frames)
    tensor = build_tensor(frames, pid_lookup)
    np.save(args.out, tensor)
    print(
        f"Tensor saved to {args.out} with shape {tensor.shape} and "
        f"{tensor.sum()} ones set."
    )


if __name__ == "__main__":
    main()