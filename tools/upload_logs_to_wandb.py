#!/usr/bin/env python3
"""Upload parsed experiment logs to Weights & Biases."""

from __future__ import annotations

import argparse
import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import wandb

METRIC_RE = re.compile(r"([A-Za-z]+@\d+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))")
PARAM_RE = re.compile(r"Parameters:?\s*(\[[^\]]*\])=\(([^)]*)\)")
TRAIN_RE = re.compile(
    r"epoch\s+(\d+)\s+training\s+\[time:\s*([-+]?\d+(?:\.\d+)?)s,\s*train loss:\s*([-+]?\d+(?:\.\d+)?)\]"
)
EVAL_RE = re.compile(
    r"epoch\s+(\d+)\s+evaluating\s+\[time:\s*([-+]?\d+(?:\.\d+)?)s,\s*valid_score:\s*([-+]?\d+(?:\.\d+)?)\]"
)
MONTHS = {
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
}


def iter_files(log_dir: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    globber = log_dir.rglob if recursive else log_dir.glob
    for path in sorted(globber(pattern)):
        if path.is_file():
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload raw log files and parsed experiment results to W&B."
    )
    parser.add_argument("--log-dir", default="log", help="Directory containing log files.")
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Glob pattern to upload.",
    )
    parser.add_argument(
        "--project",
        default="AlignRec",
        help="W&B project name. Defaults to AlignRec.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="Optional W&B entity/team. If omitted, W&B uses the logged-in default.",
    )
    parser.add_argument(
        "--run-name",
        default="upload-log-files",
        help="Name for the W&B run that records the upload.",
    )
    parser.add_argument(
        "--artifact-name",
        default="alignrec-log-files",
        help="Artifact name to create or version.",
    )
    parser.add_argument(
        "--artifact-type",
        default="logs",
        help="Artifact type shown in W&B.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only match files directly inside --log-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matched files without uploading.",
    )
    return parser.parse_args()


def strip_log_prefix(line: str) -> str:
    return re.sub(r"^[A-Z][a-z]{2}\s+.*?\s+INFO\s+", "", line.rstrip("\n")).strip(" ")


def parse_value(value: str) -> Any:
    raw_value = value
    value = value.strip()
    if not value and raw_value:
        return raw_value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        if value == "True":
            return True
        if value == "False":
            return False
        if value == "None":
            return None
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


def flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in values.items()}


def parse_metrics(text: str) -> dict[str, float]:
    return {name.lower(): float(value) for name, value in METRIC_RE.findall(text)}


def parse_parameters(text: str) -> dict[str, Any]:
    match = PARAM_RE.search(text)
    if not match:
        return {}
    try:
        names = ast.literal_eval(match.group(1))
        values = ast.literal_eval(f"({match.group(2)},)")
    except (ValueError, SyntaxError):
        return {}
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    return dict(zip(names, values))


def parse_filename_metadata(path: Path) -> dict[str, Any]:
    stem = path.stem
    parts = stem.split("-")
    metadata: dict[str, Any] = {
        "filename": path.name,
        "filename_stem": stem,
    }

    has_timestamp = (
        len(parts) >= 8
        and parts[-6] in MONTHS
        and parts[-5].isdigit()
        and parts[-4].isdigit()
        and parts[-3].isdigit()
        and parts[-2].isdigit()
        and parts[-1].isdigit()
    )
    if not has_timestamp:
        if len(parts) >= 2:
            metadata["filename_model"] = parts[0]
            metadata["filename_dataset"] = parts[1]
        return metadata

    model = "-".join(parts[:-7])
    dataset = parts[-7]
    timestamp_text = "-".join(parts[-6:])
    metadata.update(
        {
            "filename_model": model,
            "filename_dataset": dataset,
            "filename_timestamp": timestamp_text,
        }
    )
    try:
        timestamp = datetime.strptime(timestamp_text, "%b-%d-%Y-%H-%M-%S")
    except ValueError:
        return metadata

    metadata["filename_timestamp_iso"] = timestamp.isoformat()
    metadata["filename_year"] = timestamp.year
    metadata["filename_month"] = timestamp.month
    metadata["filename_day"] = timestamp.day
    metadata["filename_hour"] = timestamp.hour
    metadata["filename_minute"] = timestamp.minute
    metadata["filename_second"] = timestamp.second
    return metadata


def parse_log(path: Path) -> dict[str, Any]:
    raw_lines = path.read_text(errors="replace").splitlines()
    lines = [strip_log_prefix(line) for line in raw_lines]

    config: dict[str, Any] = {}
    dataset_stats: dict[str, dict[str, Any]] = {}
    epoch_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    current_epoch: dict[tuple[int, int], dict[str, Any]] = {}
    current_params: dict[str, Any] = {}
    trial_index = 0
    pending_metric_kind: str | None = None
    stats_section = "dataset"
    best_valid: dict[str, float] = {}
    best_test: dict[str, float] = {}
    all_over_valid: dict[str, float] = {}
    all_over_test: dict[str, float] = {}
    all_over_params: dict[str, Any] = {}
    in_all_over = False

    for index, line in enumerate(lines):
        if not line:
            continue

        if "=" in line and not line.startswith(("====", "========", "Parameters:")):
            key, value = line.split("=", 1)
            key = key.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                config[key] = parse_value(value)
                continue

        if "====Training====" in line:
            stats_section = "training"
            continue
        if "====Validation====" in line:
            stats_section = "validation"
            continue
        if "====Testing====" in line:
            stats_section = "testing"
            continue

        stat_match = re.match(r"(The number of users|Average actions of users|The number of items|Average actions of items|The number of inters|The sparsity of the dataset):\s*(.*)", line)
        if stat_match:
            section = dataset_stats.setdefault(stats_section, {})
            key = stat_match.group(1).lower().replace("the ", "").replace(" ", "_")
            section[key] = parse_value(stat_match.group(2).rstrip("%"))
            continue

        params = parse_parameters(line)
        if params:
            current_params = params
            if "=========" in line or re.search(r"\d+/\d+", line):
                trial_index += 1
            if in_all_over:
                all_over_params = params
            continue

        train_match = TRAIN_RE.search(line)
        if train_match:
            epoch = int(train_match.group(1))
            row = current_epoch.setdefault(
                (trial_index, epoch), {"epoch": epoch, "trial_index": trial_index}
            )
            row.update(flatten("params", current_params))
            row["train/time_sec"] = float(train_match.group(2))
            row["train/loss"] = float(train_match.group(3))
            continue

        eval_match = EVAL_RE.search(line)
        if eval_match:
            epoch = int(eval_match.group(1))
            row = current_epoch.setdefault(
                (trial_index, epoch), {"epoch": epoch, "trial_index": trial_index}
            )
            row.update(flatten("params", current_params))
            row["eval/time_sec"] = float(eval_match.group(2))
            row["valid/score"] = float(eval_match.group(3))
            continue

        if line.startswith("valid result:"):
            metrics = parse_metrics(line)
            if metrics:
                target = all_over_valid if in_all_over else best_valid
                target.update(metrics)
                pending_metric_kind = None
            else:
                pending_metric_kind = "valid"
            continue

        if line.startswith("test result:"):
            metrics = parse_metrics(line)
            if metrics:
                target = all_over_test if in_all_over else best_test
                target.update(metrics)
                pending_metric_kind = None
            else:
                pending_metric_kind = "test"
            continue

        if line.startswith("best valid result:"):
            best_valid = parse_metrics(line)
            continue

        if line.startswith("best test:") or line.startswith("best test result:"):
            all_over_test = parse_metrics(line)
            continue

        if line.startswith("best valid:"):
            all_over_valid = parse_metrics(line)
            continue

        if "All Over" in line:
            in_all_over = True
            continue

        if pending_metric_kind:
            metrics = parse_metrics(line)
            if metrics:
                key = max(current_epoch) if current_epoch else None
                if key is not None:
                    current_epoch[key].update(flatten(pending_metric_kind, metrics))
                pending_metric_kind = None
            continue

        if "Current BEST" in line or " BEST " in line:
            sweep_row = {"trial_index": trial_index}
            sweep_row.update(flatten("params", current_params))
            if best_valid:
                sweep_row.update(flatten("best_valid", best_valid))
            if best_test:
                sweep_row.update(flatten("best_test", best_test))
            if len(sweep_row) > 1:
                sweep_rows.append(sweep_row)

    epoch_rows = [current_epoch[key] for key in sorted(current_epoch)]

    if all_over_valid or all_over_test or all_over_params:
        all_over_row = {}
        all_over_row.update(flatten("params", all_over_params or current_params))
        all_over_row.update(flatten("all_over_best_valid", all_over_valid))
        all_over_row.update(flatten("all_over_best_test", all_over_test))
        sweep_rows.append(all_over_row)

    return {
        "config": config,
        "dataset_stats": dataset_stats,
        "epoch_rows": epoch_rows,
        "sweep_rows": sweep_rows,
        "best_valid": best_valid,
        "best_test": best_test,
        "all_over_params": all_over_params,
        "all_over_valid": all_over_valid,
        "all_over_test": all_over_test,
        "line_count": len(raw_lines),
    }


def table_from_rows(rows: list[dict[str, Any]]) -> wandb.Table | None:
    if not rows:
        return None
    columns = sorted({key for row in rows for key in row})
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(*[row.get(column) for column in columns])
    return table


def upload_one_log(path: Path, log_dir: Path, args: argparse.Namespace) -> None:
    parsed = parse_log(path)
    filename_metadata = parse_filename_metadata(path)
    rel_path = path.relative_to(log_dir)
    run_name = f"{args.run_name}-{path.stem}" if args.run_name else path.stem
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        job_type="upload_parsed_log",
        config={
            **filename_metadata,
            **parsed["config"],
            "source_log": str(rel_path),
            "source_log_lines": parsed["line_count"],
            **flatten("dataset_stats/dataset", parsed["dataset_stats"].get("dataset", {})),
            **flatten("dataset_stats/training", parsed["dataset_stats"].get("training", {})),
            **flatten("dataset_stats/validation", parsed["dataset_stats"].get("validation", {})),
            **flatten("dataset_stats/testing", parsed["dataset_stats"].get("testing", {})),
        },
        reinit=True,
    )

    artifact = wandb.Artifact(f"{args.artifact_name}-{path.stem}", type=args.artifact_type)
    artifact.add_file(str(path), name=str(rel_path))
    run.log_artifact(artifact)

    for row_number, row in enumerate(parsed["epoch_rows"]):
        run.log(row, step=row_number)

    summary_rows = {}
    summary_rows.update(flatten("best_valid", parsed["best_valid"]))
    summary_rows.update(flatten("best_test", parsed["best_test"]))
    summary_rows.update(flatten("all_over_params", parsed["all_over_params"]))
    summary_rows.update(flatten("all_over_best_valid", parsed["all_over_valid"]))
    summary_rows.update(flatten("all_over_best_test", parsed["all_over_test"]))
    for key, value in summary_rows.items():
        run.summary[key] = value

    tables = {}
    epoch_table = table_from_rows(parsed["epoch_rows"])
    sweep_table = table_from_rows(parsed["sweep_rows"])
    if epoch_table is not None:
        tables["parsed_epoch_metrics"] = epoch_table
    if sweep_table is not None:
        tables["parsed_best_and_all_over"] = sweep_table
    if tables:
        run.log(tables)

    run.finish()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist: {log_dir}")

    files = list(iter_files(log_dir, args.pattern, recursive=not args.no_recursive))
    if not files:
        raise SystemExit(f"No files matched {args.pattern!r} under {log_dir}")

    print(f"Matched {len(files)} file(s):")
    for path in files:
        parsed = parse_log(path)
        filename_metadata = parse_filename_metadata(path)
        dataset = parsed["config"].get("dataset", "unknown")
        model = parsed["config"].get("model", "unknown")
        print(
            f"{path.relative_to(log_dir)}"
            f" | model={model}"
            f" dataset={dataset}"
            f" filename_model={filename_metadata.get('filename_model', 'unknown')}"
            f" filename_dataset={filename_metadata.get('filename_dataset', 'unknown')}"
            f" filename_timestamp={filename_metadata.get('filename_timestamp', 'unknown')}"
            f" epochs={len(parsed['epoch_rows'])}"
            f" all_over={bool(parsed['all_over_valid'] or parsed['all_over_test'])}"
        )

    if args.dry_run:
        return

    for path in files:
        upload_one_log(path, log_dir, args)


if __name__ == "__main__":
    main()
