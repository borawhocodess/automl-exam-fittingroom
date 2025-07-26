from __future__ import annotations

import argparse
import json
from pathlib import Path

from fittingroom.utils import get_default_constant

BASELINES = {
    "bike_sharing_demand": 0.9457,
    "brazilian_houses": 0.9896,
    "superconductivity": 0.9311,
    "wine_quality": 0.4410,
    "yprop_4_1": 0.0778,
}


def safe_float(val):
    try:
        return None if val is None else float(val)
    except (TypeError, ValueError):
        return None


def load_metadata(path):
    with path.open("r") as f:
        return json.load(f)


def filter_by_task(metadata, task):
    if task is None:
        return metadata
    return {k: v for k, v in metadata.items() if v.get("task") == task}


def latest_rows(metadata, n, metric_key):
    rows = []
    for key in sorted(metadata.keys())[-n:]:
        e = metadata[key]
        rows.append(
            (
                e.get("task", key),
                safe_float(e.get(metric_key)),
                safe_float(e.get("fit_duration")),
                safe_float(e.get("predict_duration")),
                e.get("out_filename"),
            )
        )
    return rows


def best_rows(metadata, metric_key):
    if not metadata:
        return []

    if len({e.get("task") for e in metadata.values()}) == 1:
        best_entry = max(
            metadata.values(),
            key=lambda e: safe_float(e.get(metric_key)) or float("-inf"),
        )
        return [
            (
                best_entry.get("task"),
                safe_float(best_entry.get(metric_key)),
                safe_float(best_entry.get("fit_duration")),
                safe_float(best_entry.get("predict_duration")),
                best_entry.get("out_filename"),
            )
        ]

    best = {}
    for e in metadata.values():
        task = e.get("task")
        score = safe_float(e.get(metric_key))
        if task is None or score is None:
            continue
        if task not in best or score > safe_float(best[task].get(metric_key)):
            best[task] = e

    rows = []
    for task, e in best.items():
        rows.append(
            (
                task,
                safe_float(e.get(metric_key)),
                safe_float(e.get("fit_duration")),
                safe_float(e.get("predict_duration")),
                e.get("out_filename"),
            )
        )
    rows.sort(key=lambda r: r[0])
    return rows


def print_table(rows, show_baseline=True, show_other=True):
    long_w = get_default_constant("LONG_TABLE_CELL_WIDTH")
    col_w = get_default_constant("DEFAULT_TABLE_CELL_WIDTH")
    prec = get_default_constant("PRECISION")
    file_w = max((len(r[4] or "") for r in rows), default=0) + 1 if show_other else 0

    def fmt(x):
        return f"{x:.{prec}f}" if x is not None else "-"

    if not rows:
        return

    header = f"{'task':<{long_w}} "
    header += f"{'r2 ours':<{col_w}} "
    if show_baseline:
        header += f"{'baseline':<{col_w}} "
    if show_other:
        header += f"{'t fit':<{col_w}} {'t predict':<{col_w}} {'filename':<{file_w}}"
    print(header)
    for task, r2, t_fit, t_pred, fname in rows:
        baseline_val = safe_float(BASELINES.get(task))
        row = f"{task:<{long_w}} "
        row += f"{fmt(r2):<{col_w}} "
        if show_baseline:
            row += f"{fmt(baseline_val):<{col_w}} "
        if show_other:
            row += (
                f"{fmt(t_fit):<{col_w}} "
                f"{fmt(t_pred):<{col_w}} "
                f"{(fname or '-'): <{file_w}}"
            )
        print(row)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "-n",
        "--latest",
        type=int,
        default=5,
        metavar="N",
        help="Show the latest N runs (default: %(default)s)",
    )
    p.add_argument(
        "-t",
        "--task",
        metavar="TASK",
        help="Filter results to a single task name",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Include baseline column (omit flag to hide it)",
    )
    p.add_argument(
        "--other",
        action="store_true",
        help="Include t_fit, t_predict, and filename columns (omit flag to hide them)",
    )
    args = p.parse_args()
    show_baseline = args.baseline
    show_other = args.other

    metadata_dir = Path(get_default_constant("METADATA_DIR"))
    metadata_filename = get_default_constant("METADATA_FILENAME")
    metadata_path = metadata_dir / metadata_filename

    data = load_metadata(metadata_path)

    data = filter_by_task(data, args.task)

    latest_title = (
        f"Latest {args.latest} runs"
        if args.task is None
        else f"Latest {args.latest} runs for task '{args.task}'"
    )
    best_title = (
        "Best result per task"
        if args.task is None
        else f"Best result for task '{args.task}'"
    )

    print()
    print()
    print(latest_title)
    print()
    print_table(latest_rows(data, args.latest, "r2_test"), show_baseline, show_other)
    print()
    print()
    print(best_title)
    print()
    print_table(best_rows(data, "r2_test"), show_baseline, show_other)
    print()
    print()


if __name__ == "__main__":
    main()
