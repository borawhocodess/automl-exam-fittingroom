import datetime
import json
import os
import random
import time
from webbrowser import get

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch


def get_default_constant(name):
    """
    returns a predefined constant value for the given name
    """
    defaults = {
        "SEED": 42,
        "DEVICE": "cpu",
        "TEST_SIZE": 0.2,
        "PRECISION": 4,
        "INDENT": 4,
        "BENCHMARK_FILEPATH": "../benchmarks/fittingroom_benchmark.json",
        "LONG_TABLE_CELL_WIDTH": 20,
        "DEFAULT_TABLE_CELL_WIDTH": 10,
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

    ax.axhline(0, color=origin_line_color, linewidth=origin_line_width, alpha=origin_line_alpha)
    ax.axvline(0, color=origin_line_color, linewidth=origin_line_width, alpha=origin_line_alpha)

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


