#!/usr/bin/env python3
"""Filter SMORE-grocery log to only params used in SMORE-game."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

PARAM_RE = re.compile(r"Parameters:\s*\[(.*?)\]\s*=\((.*?)\)")
HEADER_PARAM_RE = re.compile(r"^([A-Za-z0-9_]+)=(.*)$")
BLOCK_LINE_RE = re.compile(r"(=+)\d+/\d+:")
VALID20_RE = re.compile(r"recall@20:\s*([0-9.]+)")


def parse_param_tuple(line: str):
    match = PARAM_RE.search(line)
    if not match:
        return None, None
    param_names_raw = match.group(1)
    param_values_raw = match.group(2)
    # Convert to Python objects.
    param_names = ast.literal_eval("[" + param_names_raw + "]")
    param_values = ast.literal_eval("(" + param_values_raw + ")")
    return param_names, param_values


def extract_header_values(lines: list[str]):
    header = {}
    for line in lines:
        line = line.rstrip("\n")
        m = HEADER_PARAM_RE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        header[key] = value
    return header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game-log",
        default="log/SMORE-game-Feb-02-2026-02-49-55.log",
    )
    parser.add_argument(
        "--grocery-log",
        default="log/SMORE-office-Feb-01-2026-07-37-19.log",
    )
    parser.add_argument(
        "--out-log",
        default="log/SMORE-office-parsing.log",
    )
    args = parser.parse_args()

    game_path = Path(args.game_log)
    grocery_path = Path(args.grocery_log)
    out_path = Path(args.out_log)

    game_lines = game_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    grocery_lines = grocery_path.read_text(encoding="utf-8", errors="replace").splitlines(True)

    # Identify header ranges (before first parameter line).
    def header_end(lines):
        for i, line in enumerate(lines):
            if PARAM_RE.search(line):
                return i
        return len(lines)

    game_header_end = header_end(game_lines)
    grocery_header_end = header_end(grocery_lines)

    game_header = extract_header_values(game_lines[:game_header_end])
    replace_keys = {
        "seed",
        "n_ui_layers",
        "image_knn_k",
        "text_knn_k",
        "reg_weight",
        "dropout_rate",
        "hyper_parameters",
    }

    # Collect parameter tuples used in game.
    game_param_tuples = set()
    for line in game_lines:
        _, param_values = parse_param_tuple(line)
        if param_values is not None:
            game_param_tuples.add(tuple(param_values))

    # Find summary section if present.
    summary_start = None
    for i in range(grocery_header_end, len(grocery_lines)):
        if "All Over" in grocery_lines[i]:
            summary_start = i
            break

    best_header_idx = None
    if summary_start is None:
        for i in range(len(grocery_lines) - 1, grocery_header_end - 1, -1):
            if "BEST" in grocery_lines[i] and "████" in grocery_lines[i]:
                best_header_idx = i
                break

        if best_header_idx is not None:
            def is_summary_line(line: str) -> bool:
                stripped = line.strip()
                if stripped == "":
                    return True
                if stripped.startswith("best valid:") or stripped.startswith("best test:"):
                    return True
                return "Parameters:" in line

            j = best_header_idx - 1
            while j >= grocery_header_end and is_summary_line(grocery_lines[j]):
                j -= 1
            summary_start = j + 1

    main_lines = grocery_lines[grocery_header_end:summary_start] if summary_start else grocery_lines[grocery_header_end:]
    summary_lines = grocery_lines[summary_start:] if summary_start else []

    # Split grocery log into blocks.
    blocks = []
    current_block = None
    current_params = None
    for line in main_lines:
        param_names, param_values = parse_param_tuple(line)
        if param_values is not None:
            # Start new block
            if current_block is not None:
                blocks.append((current_params, current_block))
            current_block = [line]
            current_params = tuple(param_values)
        else:
            if current_block is None:
                # stray lines after header but before first block
                current_block = [line]
                current_params = None
            else:
                current_block.append(line)

    if current_block is not None:
        blocks.append((current_params, current_block))

    # Filter blocks to only those in game params.
    kept_blocks = [(p, b) for (p, b) in blocks if p in game_param_tuples]

    total = len(kept_blocks)

    # Build output lines with updated header.
    out_lines = []
    for line in grocery_lines[:grocery_header_end]:
        m = HEADER_PARAM_RE.match(line.rstrip("\n"))
        if m and m.group(1) in replace_keys and m.group(1) in game_header:
            out_lines.append(f"{m.group(1)}={game_header[m.group(1)]}\n")
        else:
            out_lines.append(line)

    # Ensure a blank line between header and blocks if it existed in original.
    # (Keep original spacing as-is.)
    idx = 0
    for params, block in kept_blocks:
        idx += 1
        for line in block:
            # Skip parameter lines that refer to combos outside the game grid.
            if "Parameters:" in line:
                _, line_params = parse_param_tuple(line)
                if line_params is not None and tuple(line_params) not in game_param_tuples:
                    continue
            if PARAM_RE.search(line):
                line = BLOCK_LINE_RE.sub(lambda m: f"{m.group(1)}{idx}/{total}:", line, count=1)
            out_lines.append(line)

    # Rebuild summary section with filtered params and recomputed BEST.
    if summary_lines:
        out_summary_lines = []
        start_idx = 0
        if "All Over" in summary_lines[0]:
            out_summary_lines.append(summary_lines[0])
            start_idx = 1

        kept_entries = []
        i = start_idx
        best_header_line = None
        while i < len(summary_lines):
            line = summary_lines[i]
            if "BEST" in line and "████" in line:
                best_header_line = line
                break
            if "Parameters:" in line:
                _, param_values = parse_param_tuple(line)
                if param_values is not None and i + 2 < len(summary_lines):
                    valid_line = summary_lines[i + 1]
                    test_line = summary_lines[i + 2]
                    params_tuple = tuple(param_values)
                    if params_tuple in game_param_tuples:
                        out_summary_lines.extend([line, valid_line, test_line])
                        kept_entries.append((params_tuple, line, valid_line, test_line))
                    i += 3
                    continue
            out_summary_lines.append(line)
            i += 1

        # Compute BEST within kept entries (by valid recall@20).
        best_entry = None
        best_valid20 = None
        for params_tuple, param_line, valid_line, test_line in kept_entries:
            m = VALID20_RE.search(valid_line)
            if not m:
                continue
            val20 = float(m.group(1))
            if best_valid20 is None or val20 > best_valid20:
                best_valid20 = val20
                best_entry = (param_line, valid_line, test_line)

        if best_entry:
            if out_summary_lines and out_summary_lines[-1].strip() != "":
                out_summary_lines.append("\n")
            if best_header_line:
                out_summary_lines.append(best_header_line)
            else:
                out_summary_lines.append("█████████████ BEST ████████████████\n")

            best_param_line, best_valid_line, best_test_line = best_entry
            best_param_line = re.sub(r"INFO\\s+Parameters:", "INFO \\tParameters:", best_param_line)
            best_valid_line = re.sub(r"^\\s*best valid:\\s*", "Valid: ", best_valid_line)
            best_test_line = re.sub(r"^\\s*best test:\\s*", "Test: ", best_test_line)
            best_valid_line = best_valid_line.replace(" best valid:", "Valid:").lstrip()
            best_test_line = best_test_line.replace(" best test:", "Test:").lstrip()
            out_summary_lines.extend([best_param_line, best_valid_line, best_test_line])

        out_lines.extend(out_summary_lines)

    out_path.write_text("".join(out_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
