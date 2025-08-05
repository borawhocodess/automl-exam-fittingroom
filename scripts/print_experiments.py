from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from fittingroom.utils import ColorFormatter

    _formatter = ColorFormatter("%(levelname)s:%(name)s: %(message)s")
except Exception:
    _formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")

handler = logging.StreamHandler()
handler.setFormatter(_formatter)
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger("print-exp")

ROOT = Path(__file__).absolute().resolve().parent.parent
EXP_METADATA = ROOT / "metadata" / "experiment_metadata.json"

BASELINES = {
    "bike_sharing_demand": 0.9457,
    "brazilian_houses": 0.9896,
    "superconductivity": 0.9311,
    "wine_quality": 0.4410,
    "yprop_4_1": 0.0778,
}


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--list",
        action="store_true",
    )
    g.add_argument(
        "--ids",
        metavar="IDLIST",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
    )
    return p.parse_args()


def load_metadata() -> dict:
    if not EXP_METADATA.exists():
        return {}
    try:
        with EXP_METADATA.open() as f:
            return json.load(f) or {}
    except json.JSONDecodeError:
        log.error("Could not decode JSON in %s", EXP_METADATA)
        return {}


def cmd_list():
    data = load_metadata()
    if not data:
        log.info("No experiments recorded yet.")
        return

    id_width = max(26, max(len(k) for k in data))
    preset_width = max(12, max(len(rec.get("preset", "-")) for rec in data.values()))
    header = f"{'key':<{id_width}}  {'preset':<{preset_width}}  completed  macro_r2"
    print(header)
    print("-" * len(header))

    for key, rec in sorted(data.items()):
        preset = rec.get("preset", "-")
        comp_val = rec.get("completed")
        comp = "-" if comp_val is None else str(comp_val)
        macro = rec.get("macro_avg_r2")
        macro_str = f"{macro:.4f}" if macro is not None else "-"
        print(f"{key:<{id_width}}  {preset:<{preset_width}}  {comp:<9}  {macro_str}")


def cmd_table(id_list: list[str], show_baseline: bool):
    """Render a table where rows = datasets, columns = experiments (plus baseline)."""
    data = load_metadata()

    missing = [eid for eid in id_list if eid not in data]
    if missing:
        log.error("ID(s) not found in %s: %s", EXP_METADATA, ", ".join(missing))
        return

    tasks = list(BASELINES.keys())
    header = ["task"]
    if show_baseline:
        header.append("baseline")
    header.extend(id_list)

    table = [header]

    for task in tasks:
        row = [task]
        if show_baseline:
            row.append(f"{BASELINES[task]:.4f}")
        for eid in id_list:
            rec = data[eid]
            cell = "-"
            taskinfo = rec.get("tasks", {}).get(task)
            if taskinfo is not None:
                cell = f"{taskinfo['mean']:.4f}"
            row.append(cell)
        table.append(row)

    macro_row = ["macro_avg"]
    if show_baseline:
        baseline_macro = sum(BASELINES.values()) / len(BASELINES)
        macro_row.append(f"{baseline_macro:.4f}")
    for eid in id_list:
        macro = data[eid].get("macro_avg_r2")
        macro_row.append(f"{macro:.4f}" if macro is not None else "-")
    table.append(macro_row)

    col_widths = [max(len(str(row[c])) for row in table) for c in range(len(table[0]))]

    def fmt_row(row):
        return "  ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row))

    print(fmt_row(table[0]))
    print("-" * len(fmt_row(table[0])))
    for row in table[1:]:
        print(fmt_row(row))


def main() -> None:
    args = parse_args()

    if args.list:
        cmd_list()
        return

    ids = [s.strip() for s in args.ids.split(",") if s.strip()]

    cmd_table(ids, args.baseline)


if __name__ == "__main__":
    print()
    main()
    print()
