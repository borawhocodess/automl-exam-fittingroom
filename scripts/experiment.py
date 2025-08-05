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
    "superconductivity",
    "wine_quality",
    "yprop_4_1",
    "exam_dataset",
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
        default="42,420,11",
        help="Comma-separated seeds list (default: 42,420,11)",
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

    METADIR.mkdir(parents=True, exist_ok=True)
    if EXP_METADATA.exists():
        with EXP_METADATA.open() as f:
            exp_meta = json.load(f)
    else:
        exp_meta = {}

    timestamp = get_timestamp()
    experiment_id = f"{timestamp}_{args.name}"

    exp_record = {
        "timestamp": timestamp,
        "preset": args.name,
        "extra_flags": PRESETS[args.name],
        "seeds": [int(s) for s in args.seeds.split(",") if s.strip()],
        "fold": args.fold,
        "tasks": {},
        "macro_avg_r2": None,
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ended": None,
        "total_runtime_s": None,
        "completed": False,
    }

    exp_meta[experiment_id] = exp_record
    with EXP_METADATA.open("w") as f:
        json.dump(exp_meta, f, indent=2)

    log.info("Created experiment skeleton: key=%s", experiment_id)

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
            log.info(f"- running {task}  seed={seed}")
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
                    log.info("  - r2_test = %.4f (ts = %s)", r2, rec["timestamp"])
                else:
                    log.info("  - r2_test missing (e.g. exam dataset)")

        if per_task_scores[task]:
            m = mean(per_task_scores[task])
            s = stdev(per_task_scores[task]) if len(per_task_scores[task]) > 1 else 0.0
            exp_record["tasks"][task] = {
                "mean": m,
                "std": s,
                "n": len(per_task_scores[task]),
                "run_ids": per_task_runs[task],
            }

            done_means = [v["mean"] for v in exp_record["tasks"].values()]
            exp_record["macro_avg_r2"] = mean(done_means)

            exp_meta[experiment_id] = exp_record
            with EXP_METADATA.open("w") as f:
                json.dump(exp_meta, f, indent=2)

            log.info(
                "Incrementally saved %s  (macro r2 so far = %.4f)",
                task,
                exp_record["macro_avg_r2"],
            )

    log.info("all runs finished - finalising record...")

    end_wall = datetime.now()
    exp_record["ended"] = end_wall.astimezone().isoformat(timespec="seconds")
    exp_record["total_runtime_s"] = (end_wall - start_wall).total_seconds()
    exp_record["completed"] = True

    exp_meta[experiment_id] = exp_record
    with EXP_METADATA.open("w") as f:
        json.dump(exp_meta, f, indent=2)

    log.info("DONE — final summary written to %s (key=%s)", EXP_METADATA, experiment_id)


if __name__ == "__main__":
    print()
    main()
    print()
