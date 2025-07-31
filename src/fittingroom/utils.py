import datetime
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from fittingroom.data import Dataset


def get_default_constant(name):
    """
    returns a predefined constant value for the given name
    """
    defaults = {
        "SEED": 42,
        "DEVICE": "cpu",
        "FOLD": 1,
        "TEST_SIZE": 0.2,
        "PRECISION": 4,
        "INDENT": 4,

        "HPO_N_TRIALS": 3, # TODO: make 30 again and make it configurable

        "DATA_DIR": "data",
        "PREDICTIONS_DIR": "preds",
        "METADATA_DIR": "metadata",
        "METADATA_FILENAME": "fittingroom_metadata.json",
        "LONG_TABLE_CELL_WIDTH": 20,
        "DEFAULT_TABLE_CELL_WIDTH": 12,
    }
    return defaults.get(name, None)


def get_default_device():
    device = get_default_constant("DEVICE")
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def set_randomness_seed(seed=get_default_constant("SEED")):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_and_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    duration = end - start
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return result, duration, timestamp


def apply_style_to_ax(
    ax,
    x_border,
    y_border,
    x_margin=0.002,
    y_margin=55,
    nbins=5,
    origin_line_color="black",
    origin_line_width=0.5,
    origin_line_alpha=0.1,
):
    """
    Apply consistent style:
        - centered origin
        - origin lines
        - margin
        - tick limits
    """
    x_lim = x_border + x_margin
    y_lim = y_border + y_margin

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)

    ax.axhline(
        0, color=origin_line_color, linewidth=origin_line_width, alpha=origin_line_alpha
    )
    ax.axvline(
        0, color=origin_line_color, linewidth=origin_line_width, alpha=origin_line_alpha
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins))

    return ax


def apply_style_to_figure(fig, X, y, **kwargs):
    X = X.flatten()
    y = y.flatten()

    x_border = max(abs(X.min()), abs(X.max()))
    y_border = max(abs(y.min()), abs(y.max()))

    for ax in fig.axes:
        apply_style_to_ax(ax, x_border, y_border, **kwargs)


def print_datasets_overview(
    datadir=get_default_constant("DATA_DIR"),
    tasks=[
        "bike_sharing_demand",
        "brazilian_houses",
        "superconductivity",
        "wine_quality",
        "yprop_4_1",
    ],
    fold=get_default_constant("FOLD"),
    table_cell_width=get_default_constant("DEFAULT_TABLE_CELL_WIDTH"),
):
    header_map = {
        "bike_sharing_demand": "bsd",
        "brazilian_houses": "bh",
        "superconductivity": "sc",
        "wine_quality": "wq",
        "yprop_4_1": "y41",
    }

    rows = ["X_train", "X_test", "y_train", "y_test", "has_nan", "has_cat", "has_bool"]

    width = table_cell_width

    datadirpath = Path(datadir)

    datasets = {task: Dataset.load(datadirpath, task, fold) for task in tasks}

    header = f"{'':<{width}}" + "".join(
        f"{header_map[task]:<{width}}" for task in tasks
    )
    print(header)

    for row in rows:
        line = f"{row:<{width}}"
        for task in tasks:
            data = (
                getattr(datasets[task], row)
                if row in ["X_train", "X_test", "y_train", "y_test"]
                else None
            )
            if row in ["X_train", "X_test", "y_train", "y_test"]:
                shape = str(data.shape) if data is not None else "None"
                line += f"{shape:<{width}}"
            elif row == "has_nan":
                has_nan = datasets[task].X_train.isnull().any().any()
                line += f"{str(has_nan):<{width}}"
            elif row == "has_cat":
                has_cat = any(datasets[task].X_train.dtypes == "object") or any(
                    d.name == "category" for d in datasets[task].X_train.dtypes
                )
                line += f"{str(has_cat):<{width}}"
            elif row == "has_bool":
                has_bool = any(d.name == "bool" for d in datasets[task].X_train.dtypes)
                line += f"{str(has_bool):<{width}}"
        print(line)


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
