# hyper_pram_log_0126.py
import re
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


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


def safe_sort_key(x: Any):
    # 숫자는 숫자대로, 문자열은 문자열대로 정렬되게
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))


def plot_hp_mean(df: pd.DataFrame, hp: str, metric: str, out_path: Path):
    # seed 포함 다른 파라미터 모두 "마진" 처리: hp별 metric 평균
    g = df.groupby(hp, dropna=False)[metric].mean().reset_index()
    g = g.sort_values(by=hp, key=lambda col: col.map(safe_sort_key))

    plt.figure()
    plt.plot(g[hp].values, g[metric].values, marker="o")
    plt.xlabel(hp)
    plt.ylabel(metric)
    plt.title(f"{metric} mean by {hp} (avg over seeds & other params)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="valid_r20",
        choices=["valid_r20", "test_r20", "valid_n20", "test_n20"],
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    out_dir = log_path.parent / (log_path.name + "_plots_r20")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_log(log_path)
    if df.empty:
        raise RuntimeError("No experiments parsed. Check log format / regex.")

    # metric 없는 row 제거
    df = df.dropna(subset=[args.metric]).copy()

    # seed 제외하고 hp 목록 구성
    hp_cols = [c for c in df.columns if c not in {"seed", "valid_r20", "test_r20", "valid_n20", "test_n20"}]

    # CSV 저장
    csv_path = out_dir / "parsed_r20.csv"
    df.to_csv(csv_path, index=False)

    for hp in hp_cols:
        out_path = out_dir / f"{args.metric}_mean_by_{hp}.png"
        plot_hp_mean(df, hp, args.metric, out_path)

    print("[OK] parsed rows:", len(df))
    print("[OK] saved csv  :", csv_path)
    print("[OK] saved plots:", out_dir)


if __name__ == "__main__":
    main()
