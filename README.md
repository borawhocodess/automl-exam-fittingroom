# fittingroom AutoML Exam - SS25 (Tabular Data)

The purpose of this repository is to house our AutoML system as the fittingroom team for the AutoML exam in SS25.

What we did:
- ...........
- ...........


## Setup

Clone the repository, afterwards using [`uv`](https://github.com/astral-sh/uv):

```bash
uv run python download-datasets.py
```

```bash
uv pip install -e .
```


## One-click run command

This will train our AutoML system and generate predictions for `X_test`:

```bash
uv run python run.py \
  --seed 42 \
  --data-dir data \
  --task bike_sharing_demand \
  --fold 1 \
  --out-dir preds \
  --out-filename output.npy \
  --log-level debug \
  --ask-expert-opinion
```


## Other script commands

To run all tasks:

```bash
uv run python scripts/run_all_tasks.py
```

To get the latest and best results in comparison to the baselines:

```bash
uv run python scripts/print_r2_tables.py --baseline
```


## Details

...........

### Model choices and whys

- ...........

### Search spaces and strategies

- ...........

### Performance vs cost tradeoffs

- ...........

### Compute usage

- ...........


## Code

We provide the following:

- [`run.py`](./run.py): ...........
- [`download-datasets.py`](./download-datasets.py): ...........
- [`src/fittingroom/pipeline.py`](./src/fittingroom/pipeline.py): ...........
- ...........


## Performance table

TODO: eval 5 repetitions, different seeds, different folds, average, fill

| Dataset             | Baseline | Ours |
| ------------------- | -------- | ---- |
| bike_sharing_demand | 0.9457   | - |
| brazilian_houses    | 0.9896   | - |
| superconductivity   | 0.9311   | - |
| wine_quality        | 0.4410   | - |
| yprop_4_1           | 0.0778   | - |
| **exam dataset**    | -        | **-** |

The scores listed are the R² values calculated using scikit-learn's `metrics.r2_score`.


## Thanks

A big thank you goes out to
- the team
- our families
- our friends
- dear AutoML SS25 orga team
- university of freiburg
- sklearn developers
