import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_MODEL_RE = re.compile(r"\bmodel\s*=\s*([A-Za-z0-9_\-]+)")
_MODEL_CLASS_RE = re.compile(r"\bINFO\s+([A-Za-z0-9_\-]+)\(")
_DATASET_RE = re.compile(r"\bdataset\s*=\s*([A-Za-z0-9_\-]+)")
_SAVED_MODEL_RE = re.compile(r"Saved best model to saved/([A-Za-z0-9_\-]+)_best\.pth")
_RUN_START_RE = re.compile(r"=+\s*\d+/\d+\s*:\s*Parameters", re.IGNORECASE)
_EPOCH_TIME_RE = re.compile(
    r"\bepoch\s+(?P<epoch>\d+)\s+(?:training|evaluating)\s+\[time:\s*(?P<sec>[0-9]*\.?[0-9]+)s",
    re.IGNORECASE,
)
_S_PER_EPOCH_RE = re.compile(r"\b([0-9]*\.?[0-9]+)\s*s/epoch\b", re.IGNORECASE)
_TRAINABLE_RE = re.compile(
    r"Trainable parameters:\s*([0-9,]+)", re.IGNORECASE
)

def _parse_model_name(log_path: Path) -> Optional[str]:
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _MODEL_RE.search(line)
            if m:
                return m.group(1)
            mc = _MODEL_CLASS_RE.search(line)
            if mc:
                return mc.group(1)
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _SAVED_MODEL_RE.search(line)
            if m:
                return m.group(1)
    return None


def _parse_dataset_name(log_path: Path) -> Optional[str]:
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _DATASET_RE.search(line)
            if m:
                return m.group(1)
    return None


def _parse_s_per_epoch_avg(log_path: Path) -> Optional[float]:
    per_epoch: Dict[int, float] = defaultdict(float)
    epoch_values: List[float] = []
    s_per_epoch_vals: List[float] = []
    saw_run_header = False

    def flush_run() -> None:
        if per_epoch:
            epoch_values.extend(per_epoch.values())
            per_epoch.clear()

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if _RUN_START_RE.search(line):
                saw_run_header = True
                flush_run()
                continue

            sm = _S_PER_EPOCH_RE.search(line)
            if sm:
                s_per_epoch_vals.append(float(sm.group(1)))

            m = _EPOCH_TIME_RE.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            sec = float(m.group("sec"))

            # Logs without explicit run headers can still restart epochs for each run.
            if not saw_run_header and epoch == 0 and 0 in per_epoch:
                flush_run()
            per_epoch[epoch] += sec

    flush_run()

    if epoch_values:
        return float(statistics.mean(epoch_values))
    if s_per_epoch_vals:
        return float(statistics.mean(s_per_epoch_vals))
    return None

def _parse_trainable_params(log_path: Path) -> Optional[int]:
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _TRAINABLE_RE.search(line)
            if m:
                # 쉼표 제거 후 int 변환
                return int(m.group(1).replace(",", ""))
    return None


def _collect_files(patterns: List[str], log_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.exists() and p.is_file():
            files.append(p)
            continue

        if any(ch in pat for ch in ["*", "?", "["]):
            if p.parent == Path("."):
                files.extend(sorted(log_dir.glob(pat)))
            else:
                files.extend(sorted(p.parent.glob(p.name)))
            continue

        candidate = log_dir / pat
        if candidate.exists() and candidate.is_file():
            files.append(candidate)

    uniq: List[Path] = []
    seen = set()
    for f in files:
        rp = str(f.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(f)
    return uniq


def main() -> None:
    parser = argparse.ArgumentParser(description="Print average s/epoch per model from log files.")
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--logs", dest="logs_opt", nargs="+")
    parser.add_argument("--log-dir", default="./log")
    args = parser.parse_args()

    log_patterns = (args.logs_opt or []) + (args.logs or [])
    if not log_patterns:
        print("No log files matched.")
        return

    log_dir = Path(args.log_dir)
    files = _collect_files(log_patterns, log_dir)
    if not files:
        print("No log files matched.")
        return

    model_to_vals: Dict[str, List[float]] = defaultdict(list)
    model_to_params: Dict[str, Optional[int]] = {}

    for f in files:
        avg = _parse_s_per_epoch_avg(f)
        if avg is None:
            continue

        model = _parse_model_name(f) or f.stem
        dataset = _parse_dataset_name(f)
        key = f"{model}-{dataset}" if dataset else model

        model_to_vals[key].append(avg)

        # ✅ trainable parameter 파싱
        params = _parse_trainable_params(f)
        if params is not None:
            model_to_params[key] = params

    if not model_to_vals:
        print("No s/epoch info found in matched logs.")
        return

    for model in sorted(model_to_vals.keys()):
        vals = model_to_vals[model]
        overall = float(statistics.mean(vals))
        params = model_to_params.get(model)

        if params is not None:
            print(
                f"\t{model}: {overall:.4f} s/epoch\t"
                f"\t| trainable parameters: {params:,}"
            )
        else:
            print(
                f"\t{model}: {overall:.4f} s/epoch\t"
                f"\t| trainable parameters: N/A"
            )

if __name__ == "__main__":
    main()
