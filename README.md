# fittingroom AutoML Exam - SS25 (Tabular Data)

the purpose of this repository is to house our AutoML system as the fittingroom team for the AutoML exam in SS25.

what we did:
- ...........
- ...........


## Installation

Clone the repository, afterwards install dependencies via:
```
pip install -e .
```

## One-click command

This will train our AutoML system and generate predictions for `X_test`:

```
python oneclick.py \
  --task bike_sharing_demand \
  --seed 42 \
  --fold 0 \
  --datadir data \
  --output-path preds/bike_sharing_demand/output.npy \
  --log-level info
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

- [`oneclick.py`](./oneclick.py): ...........
- [`src/fittingroom/automl.py`](./src/fittingroom/automl.py): ...........
- [`src/fittingroom/data.py`](./src/fittingroom/data.py): ...........
- [`src/fittingroom/pipeline.py`](./src/fittingroom/pipeline.py): ...........
- ...........


## Reference performance

| Dataset | Baseline | Ours |
| -- | -- | -- |
| bike_sharing_demand | 0.9457 | - |
| brazilian_houses | 0.9896 | - |
| superconductivity | 0.9311 | - |
| wine_quality | 0.4410 | - |
| yprop_4_1 | 0.0778 | - |

The scores listed are the R² values calculated using scikit-learn's `metrics.r2_score`.

## Thanks

a big thank you goes out to
- the team
- our families
- our friends
- dear AutoML SS25 orga team
- university of freiburg
- sklearn developers

