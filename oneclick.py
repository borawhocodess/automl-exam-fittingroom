from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from fittingroom.automl import AutoML
from fittingroom.data import Dataset

logger = logging.getLogger(__name__)

FILE = Path(__file__).absolute().resolve()
DATADIR = FILE.parent / "data"


def main(
    seed: int,
    output_path: Path,
    datadir: Path,
    task: str,
    fold: int,
):
    dataset = Dataset.load(datadir=datadir, task=task, fold=fold)

    logger.info("Fitting AutoML")

    automl = AutoML(seed=seed)

    automl.fit(dataset.X_train, dataset.y_train)

    test_preds: np.ndarray = automl.predict(dataset.X_test)

    logger.info("Writing predictions to disk")

    with output_path.open("wb") as f:
        np.save(f, test_preds)

    if dataset.y_test is not None:
        r2_test = r2_score(dataset.y_test, test_preds)
        logger.info(f"R^2 on test set: {r2_test}")
    else:
        # setting for the exam dataset
        logger.info(f"No test set for task '{task}'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/bike_sharing_demand/1/predictions.npy"),
        help=(
            "The path to save the predictions to."
        ),
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
        help=(
            "The directory where the datasets are stored."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help=(
            "The fold to run on."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Whether to log only warnings and errors."
        ),
    )

    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running task {args.task}"
        f"\n{args}"
    )

    main(
        seed=args.seed,
        output_path=args.output_path,
        task=args.task,
        datadir=args.datadir,
        fold=args.fold,
    )
