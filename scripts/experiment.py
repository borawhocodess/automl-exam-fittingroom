from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

try:
    from fittingroom.utils import get_timestamp
except Exception:
    get_timestamp = lambda: datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    from fittingroom.utils import ColorFormatter

    _formatter = ColorFormatter("%(levelname)s:%(name)s: %(message)s")
except Exception:
    _formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")

handler = logging.StreamHandler()
handler.setFormatter(_formatter)
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger("experiment")

FILE = Path(__file__).absolute().resolve()
ROOT = FILE.parent.parent
DATADIR = (ROOT / "data").resolve()
METADIR = (ROOT / "metadata").resolve()
FIT_METADATA = METADIR / "fittingroom_metadata.json"
EXP_METADATA = METADIR / "experiment_metadata.json"

PRESETS = {
    "plain-hb": [
        "--hpo-method",
        "hyperband",
    ],
    "post-preds-hb": [
        "--hpo-method",
        "hyperband",
        "--add-post-hpo-preds-as-features",
    ],
    "ours-hb": [
        "--hpo-method",
        "hyperband",
        "--add-post-hpo-preds-as-features",
        "--use-bo-tabpfn-surrogate",
    ],
    "default-preds-hb": [
        "--hpo-method",
        "hyperband",
        "--add-default-preds-as-features",
    ],
}

TASKS = [
    "bike_sharing_demand",
    "brazilian_houses",
    # "superconductivity",
    # "wine_quality",
    # "yprop_4_1",
    # "exam_dataset",
]


def parse_args():
    """
    parse em up
    """
    p = argparse.ArgumentParser()

    p.add_argument(
        "--name",
        required=True,
        help=f"Preset experiment to run (one of: {', '.join(PRESETS)})",
    )
    p.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds list (default: 0,1,2)",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold to run (default: 1)",
    )
    p.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Verbosity",
    )
    return p.parse_args()


def build_cmd(task: str, seed: int, fold: int, extra_flags: list[str]) -> list[str]:
    """
    command
    """
    return [
        "python",
        "run.py",
        "--task",
        task,
        "--seed",
        str(seed),
        "--fold",
        str(fold),
        "--datadir",
        str(DATADIR),
        "--log-level",
        "info",
        *extra_flags,
    ]


def latest_record(task: str, seed: int, fold: int) -> dict | None:
    """
    newest metadata lookup
    """
    if not FIT_METADATA.exists():
        log.error("Expected %s but it does not exist", FIT_METADATA)
        return None

    with FIT_METADATA.open() as f:
        meta = json.load(f)

    # entries are keyed by timestamp string; take the newest that matches
    matches = [
        rec
        for rec in meta.values()
        if rec.get("task") == task
        and rec.get("seed") == seed
        and rec.get("fold") == fold
    ]
    return sorted(matches, key=lambda r: r["timestamp"])[-1] if matches else None


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    if args.name not in PRESETS:
        log.error(f"Unknown preset '{args.name}'. Available: {', '.join(PRESETS)}")
        raise SystemExit(1)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    preset_flags = PRESETS[args.name]

    log.info(f"Preset   : {args.name}  →  {preset_flags}")
    log.info(f"Seeds    : {seeds}")
    log.info(f"Fold     : {args.fold}")

    start_wall = datetime.now()

    per_task_scores: dict[str, list[float]] = {t: [] for t in TASKS}
    per_task_runs: dict[str, dict[str, float]] = {t: {} for t in TASKS}

    for task in TASKS:
        for seed in seeds:
            cmd = build_cmd(task, seed, args.fold, preset_flags)
            log.info(f"▶ running {task}  seed={seed}")
            log.debug("Full cmd: %s", shlex.join(cmd))

            res = subprocess.run(cmd, cwd=ROOT)  # inherit stdout/stderr
            if res.returncode != 0:
                log.error("Run failed (exit %s) — aborting experiment.", res.returncode)
                raise SystemExit(res.returncode)

            rec = latest_record(task, seed, args.fold)
            if rec is None:
                log.warning("No metadata found for %s seed=%s", task, seed)
            else:
                r2 = rec.get("r2_test")
                if r2 is not None:
                    per_task_scores[task].append(r2)
                    per_task_runs[task][rec["timestamp"]] = r2
                    log.info("  ↳ r2_test = %.4f (ts = %s)", r2, rec["timestamp"])
                else:
                    log.info("  ↳ r2_test missing (e.g. exam dataset)")

    end_wall = datetime.now()
    duration_s = (end_wall - start_wall).total_seconds()

    log.info("all runs finished in: %.1f s", duration_s)

    log.info("aggregating results…")

    agg = {}

    for task, scores in per_task_scores.items():
        if scores:
            m = mean(scores)
            s = stdev(scores) if len(scores) > 1 else 0.0
            agg[task] = {
                "mean": m,
                "std": s,
                "n": len(scores),
                "run_ids": per_task_runs[task],
            }
            log.info("  %s : %.4f ± %.4f  (n=%d)", task, m, s, len(scores))
        else:
            log.info("  %s : no numeric scores collected", task)

    numeric_means = [v["mean"] for v in agg.values() if v]
    macro_avg = mean(numeric_means) if numeric_means else None
    log.info(
        "Macro-average r2 across tasks = %s",
        f"{macro_avg:.4f}" if macro_avg is not None else "N/A",
    )

    log.info("Step 4 complete — metrics aggregated.")

    end_wall = datetime.now()
    runtime_s = (end_wall - start_wall).total_seconds()

    timestamp = get_timestamp()

    experiment_id = f"{timestamp}_{args.name}"
    exp_record = {
        "timestamp": timestamp,
        "preset": args.name,
        "extra_flags": preset_flags,
        "seeds": seeds,
        "fold": args.fold,
        "tasks": agg,  # per-task mean/std/n
        "macro_avg_r2": macro_avg,
        "started": start_wall.astimezone().isoformat(timespec="seconds"),
        "ended": end_wall.astimezone().isoformat(timespec="seconds"),
        "total_runtime_s": runtime_s,
    }

    # --- write to experiment_metadata.json ---
    METADIR.mkdir(parents=True, exist_ok=True)
    if EXP_METADATA.exists():
        with EXP_METADATA.open() as f:
            exp_meta = json.load(f)
    else:
        exp_meta = {}

    exp_meta[experiment_id] = exp_record

    with EXP_METADATA.open("w") as f:
        json.dump(exp_meta, f, indent=2)

    log.info(
        "Step 5 complete — summary written to %s (key=%s)", EXP_METADATA, experiment_id
    )


if __name__ == "__main__":
    print()
    main()
    print()
