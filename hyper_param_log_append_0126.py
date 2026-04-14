# hyper_param_log_append_0126.py
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------------
# regex patterns (네 로그 포맷에 맞춤)
# -----------------------------
RE_PARAMS = re.compile(
    r"Parameters:\s*\[([^\]]+)\]\s*=\s*\(([^)]+)\)",
    re.IGNORECASE,
)

RE_BEST_VALID_R20 = re.compile(
    r"best\s+valid\s*:\s*.*?recall@20:\s*([0-9]*\.[0-9]+)",
    re.IGNORECASE,
)

RE_BEST_TEST_R20 = re.compile(
    r"best\s+test\s*:\s*.*?recall@20:\s*([0-9]*\.[0-9]+)",
    re.IGNORECASE,
)

RE_RECALL20 = re.compile(
    r"recall@20:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)

RE_BEST_VALID_N20 = re.compile(
    r"best\s+valid\s*:\s*.*?ndcg@20:\s*([0-9]*\.[0-9]+)",
    re.IGNORECASE,
)

RE_BEST_TEST_N20 = re.compile(
    r"best\s+test\s*:\s*.*?ndcg@20:\s*([0-9]*\.[0-9]+)",
    re.IGNORECASE,
)

RE_NDCG20 = re.compile(
    r"ndcg@20:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)


def _parse_tuple_values(s: str) -> List[Any]:
    out: List[Any] = []
    for tok in s.split(","):
        tok = tok.strip()
        if re.fullmatch(r"[-+]?\d+", tok):
            out.append(int(tok))
        else:
            try:
                out.append(float(tok))
            except Exception:
                out.append(tok)
    return out


def _find_recall20(lines: List[str], start: int, end: int) -> Optional[float]:
    for k in range(start, min(end, len(lines))):
        m = RE_RECALL20.search(lines[k])
        if m:
            return float(m.group(1))
    return None


def _find_ndcg20(lines: List[str], start: int, end: int) -> Optional[float]:
    for k in range(start, min(end, len(lines))):
        m = RE_NDCG20.search(lines[k])
        if m:
            return float(m.group(1))
    return None


def parse_log(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    rows: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        m = RE_PARAMS.search(lines[i])
        if not m:
            i += 1
            continue

        keys_raw = m.group(1)
        vals_raw = m.group(2)

        keys = [k.strip().strip("'").strip('"') for k in keys_raw.split(",")]
        vals = _parse_tuple_values(vals_raw)
        if len(keys) != len(vals):
            i += 1
            continue

        row = {k: v for k, v in zip(keys, vals)}

        best_valid: Optional[float] = None
        best_test: Optional[float] = None
        best_valid_n20: Optional[float] = None
        best_test_n20: Optional[float] = None

        # 파라미터 블록 내에서 valid/test result를 추출
        j = i + 1
        pending_valid: Optional[float] = None
        pending_valid_n20: Optional[float] = None
        while j < len(lines) and not RE_PARAMS.search(lines[j]):
            line = lines[j].lower()

            # best valid/test 패턴 우선 지원
            mv = RE_BEST_VALID_R20.search(lines[j])
            if mv:
                val = float(mv.group(1))
                if best_valid is None or val > best_valid:
                    best_valid = val
            mt = RE_BEST_TEST_R20.search(lines[j])
            if mt:
                val = float(mt.group(1))
                if best_test is None or val > best_test:
                    best_test = val
            mvn = RE_BEST_VALID_N20.search(lines[j])
            if mvn:
                val = float(mvn.group(1))
                if best_valid_n20 is None or val > best_valid_n20:
                    best_valid_n20 = val
            mtn = RE_BEST_TEST_N20.search(lines[j])
            if mtn:
                val = float(mtn.group(1))
                if best_test_n20 is None or val > best_test_n20:
                    best_test_n20 = val

            if "valid result" in line:
                val = _find_recall20(lines, j, j + 4)
                if val is not None:
                    pending_valid = val
                n20 = _find_ndcg20(lines, j, j + 4)
                if n20 is not None:
                    pending_valid_n20 = n20
            if "test result" in line:
                val = _find_recall20(lines, j, j + 4)
                n20 = _find_ndcg20(lines, j, j + 4)
                if val is not None:
                    if pending_valid is not None:
                        if best_valid is None or pending_valid > best_valid:
                            best_valid = pending_valid
                            best_test = val
                            if pending_valid_n20 is not None:
                                best_valid_n20 = pending_valid_n20
                                if n20 is not None:
                                    best_test_n20 = n20
                        pending_valid = None
                        pending_valid_n20 = None
                    elif best_valid is None:
                        if best_test is None or val > best_test:
                            best_test = val
                            if n20 is not None:
                                best_test_n20 = n20

            j += 1

        # best valid/test가 없으면 스킵
        if best_valid is None and best_test is None and best_valid_n20 is None and best_test_n20 is None:
            i = j
            continue

        row["valid_r20"] = best_valid
        row["test_r20"] = best_test
        row["valid_n20"] = best_valid_n20
        row["test_n20"] = best_test_n20
        rows.append(row)

        i = j

    return pd.DataFrame(rows)


def _make_key(df: pd.DataFrame, key_cols: List[str]) -> pd.Series:
    return df[key_cols].astype(object).apply(lambda r: tuple(r.tolist()), axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default=None)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        out_dir = Path("./log/ANCHORREC-baby-Jan-20-2026-18-05-32.log_plots_r20")
        csv_path = out_dir / "parsed_r20.csv"

    new_df = parse_log(log_path)
    if new_df.empty:
        raise RuntimeError("No experiments parsed. Check log format / regex.")

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
    else:
        old_df = pd.DataFrame()

    metric_cols = {"valid_r20", "test_r20", "valid_n20", "test_n20"}
    key_cols = [c for c in new_df.columns if c not in metric_cols]

    # 컬럼 정렬 맞추기 (union)
    all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))
    old_df = old_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    if not old_df.empty:
        old_keys = set(_make_key(old_df, key_cols))
        new_keys = _make_key(new_df, key_cols)
        new_df = new_df[~new_keys.isin(old_keys)].copy()

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.to_csv(csv_path, index=False)

    print("[OK] parsed new rows:", len(new_df))
    print("[OK] total rows    :", len(combined))
    print("[OK] saved csv     :", csv_path)


if __name__ == "__main__":
    main()
