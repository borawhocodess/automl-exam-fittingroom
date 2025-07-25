from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from fittingroom import FittingRoom
from fittingroom.data import Dataset
from fittingroom.utils import get_default_constant, pxp, run_and_time


class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        formatted = super().format(record)
        prefix, _, message = formatted.partition(": ")
        return f"{color}{prefix}:{self.RESET} {message}{self.RESET}"


handler = logging.StreamHandler()
formatter = ColorFormatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logging.getLogger().handlers = [handler]
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"


def main(
    seed: int,
    output_path: Path,
    datadir: Path,
    task: str,
    fold: int,
    ask_expert_opinion: bool,
):
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info("dataset is going into the fittingroom")

    fittingroom = FittingRoom(seed=seed, ask_expert_opinion=ask_expert_opinion)

    (fittingroom, fit_duration, _) = run_and_time(
        fittingroom.fit, dataset.X_train, dataset.y_train
    )
    (test_preds, pred_duration, timestamp) = run_and_time(
        fittingroom.predict, dataset.X_test
    )

    precision = get_default_constant("PRECISION")

    logger.info(
        f"total time spent in fitting room: {fit_duration + pred_duration:.{precision}f} s"
    )

    if output_path is None:
        predictions_dir = get_default_constant("PREDICTIONS_DIR")
        predictions_dir_path = Path(predictions_dir)
        predictions_filename = f"{task}_{timestamp}_fold_{fold}.npy"
        output_path = predictions_dir_path / task / predictions_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        np.save(f, test_preds)
        logger.info(f"predictions saved to: {output_path}")

    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"r2 on test set: {r2_test:.{precision}f}")
    else:
        # setting for the exam dataset
        logger.info(f"no test set for task {task}")

    metadata_filename = get_default_constant("METADATA_FILENAME")
    metadata_dir = get_default_constant("METADATA_DIR")
    metadata_dir_path = Path(metadata_dir)
    metadata_dir_path.mkdir(parents=True, exist_ok=True)
    metadata_file_path = metadata_dir_path / metadata_filename

    metadata_entry = {
        "timestamp": timestamp,
        "seed": seed,
        "task": task,
        "fold": fold,
        "r2_test": r2_test if dataset.y_test is not None else None,
        "output_path": str(output_path),
        "fit_duration": fit_duration,
        "predict_duration": pred_duration,
    }

    if metadata_file_path.exists():
        with metadata_file_path.open("r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata[timestamp] = metadata_entry

    with metadata_file_path.open("w") as f:
        json.dump(metadata, f, indent=get_default_constant("INDENT"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=get_default_constant("SEED"),
        help=("Random seed for reproducibility"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=("The path to save the predictions to."),
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The name of the task to run on.",
        choices=[
            "bike_sharing_demand",
            "brazilian_houses",
            "superconductivity",
            "wine_quality",
            "yprop_4_1",
        ],
    )
    parser.add_argument(
        "--datadir",
        type=Path,
        default=DATADIR,
        help=("The directory where the datasets are stored."),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=get_default_constant("FOLD"),
        help=("The fold to run on."),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help=("Set the logging level."),
    )
    parser.add_argument(
        "--ask-expert-opinion",
        action="store_true",
        help=("Whether to ask for expert opinion before making choices."),
    )

    args = parser.parse_args()
    level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(level)

    pxp("entering the fittingroom")

    logger.info(f"running task {args.task}\n{args}")

    main(
        seed=args.seed,
        output_path=args.output_path,
        task=args.task,
        datadir=args.datadir,
        fold=args.fold,
        ask_expert_opinion=args.ask_expert_opinion,
    )

    pxp("left the fittingroom")
