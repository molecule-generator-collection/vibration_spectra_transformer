# Vibrational Spectra Transformer

Transformer models that predict molecular SMILES from IR and Raman spectra, along
with a fully connected baseline model for comparison. All paths are resolved
relative to the repository root.

## Setup

This project uses Python 3.12.

It targets Linux with an NVIDIA GPU (CUDA 11.8). The `Pipfile` uses the PyTorch
package index for CUDA 11.8.

```bash
pipenv install --skip-lock
```

Once the environment is set up, commands can be run with the usual `pipenv run`.

```bash
pipenv run python scripts/prepare_data.py
pipenv run python ir_raman_freq/train.py --device cuda:0
```

## 1. Prepare the Training Data

Create training, validation, and test tensors in `data_diretory/processed/` from
the included `data_diretory/size1_all.pickle` through
`data_diretory/size9_all.pickle` files.

```bash
pipenv run python scripts/prepare_data.py
```

To create a small dataset for a quick test, run:

```bash
pipenv run python scripts/prepare_data.py --max-samples 1000
```

If you change the input or output location, relative paths are still resolved
from the repository root.

```bash
pipenv run python scripts/prepare_data.py --input-dir GDB_pickle_all --output-dir data_diretory/my_dataset
```

## 2. Train a Model

```bash
pipenv run python ir_raman_freq/train.py --epochs 100 --batch-size 64
```

The initial learning rate and dropout can be adjusted from the command line.
Dropout is applied to the encoder, decoder, and SMILES embeddings.

The Transformer defaults below are shared by all three variants: Frequency + IR +
Raman, Frequency + IR, and Frequency + Raman.

| Argument | Description | Default |
|---|---|---:|
| `--initial-learning-rate` (`--learning-rate`) | Initial learning rate for AdamW | `1e-4` |
| `--dropout` | Dropout rate for the encoder, decoder, and SMILES embeddings | `0.0` |

```bash
pipenv run python ir_raman_freq/train.py \
  --initial-learning-rate 5e-5 \
  --dropout 0.1 \
  --output-dir results/lr5e-5_dropout0.1
```

`--learning-rate` and `--initial-learning-rate` are aliases for the same
argument. For comparative experiments, specify a different `--output-dir` for
each configuration to avoid overwriting models. The values actually used are
saved in `run_config.json` in the output directory and in each checkpoint.

By default, the best model is saved to
`results/ir_raman_transformer/best.pt`. A GPU is selected automatically when
CUDA is available. To select a GPU explicitly, specify `--device cuda:0`.

### Multi-GPU Training

To train on multiple GPUs on the same machine, use PyTorch's `torchrun`. For
example, to train the Frequency + IR + Raman model on four GPUs, run:

```bash
pipenv run torchrun --standalone --nproc-per-node=4 \
  ir_raman_freq/train.py \
  --epochs 100 \
  --batch-size 32 \
  --device cuda
```

Set `--nproc-per-node` to the number of GPUs to use. `--batch-size` is the batch
size per GPU, so the effective batch size in the example above is
`32 × 4 = 128`. To use only specific GPUs, restrict the visible GPUs before
launching the command.

```bash
CUDA_VISIBLE_DEVICES=0,2 pipenv run torchrun --standalone --nproc-per-node=2 \
  ir_freq/train.py --batch-size 32 --device cuda
```

The Frequency + Raman and fully connected models can be launched in the same
way.

```bash
pipenv run torchrun --standalone --nproc-per-node=2 \
  raman_freq/train.py --batch-size 32 --device cuda

pipenv run torchrun --standalone --nproc-per-node=2 \
  -m fully_connected_model.train --batch-size 32 --device cuda
```

The training data is partitioned across GPUs, and gradients are synchronized at
each step. Validation and checkpoint saving are performed only by rank 0. The
checkpoint format is the same as for single-GPU training, so the existing
`evaluate.py` and `predict.py` scripts can be used without modification. Note
that increasing the number of GPUs changes the effective batch size and may also
change the optimal learning rate.

### Automatic Optimization of the Initial Learning Rate and Dropout

You can give the same hyperparameters to all four models and use random or grid
search to find the combination that minimizes the mean validation loss across
models.

For grid search, specify multiplicative steps for the learning rate and additive
steps for dropout. For example, the following command searches the full range
using 10-fold learning-rate increments and dropout increments of 0.1:

```bash
pipenv run python scripts/optimize_hyperparameters.py \
  --search-method grid \
  --models ir_raman ir raman \
  --epochs 20 \
  --learning-rate-min 1e-5 \
  --learning-rate-max 1e-3 \
  --learning-rate-grid-factor 10 \
  --dropout-min 0.0 \
  --dropout-max 0.3 \
  --dropout-grid-step 0.1 \
  --device cuda:0
```

In this example, the learning-rate candidates are `1e-5, 1e-4, 1e-3`, and the
dropout candidates are `0.0, 0.1, 0.2, 0.3`, resulting in all 12 combinations
of their Cartesian product being tested. Because three models are specified,
the models are trained `12 combinations × 3 models = 36 times` in total. If
the upper bound of a range does not align with the step size, the upper bound is
still included as the final candidate. `--trials` is not used for grid search.

To use the original random search, specify the following. The learning rate is
sampled from a log-uniform distribution, and dropout is sampled from a uniform
distribution.

```bash
pipenv run python scripts/optimize_hyperparameters.py \
  --search-method random \
  --models ir_raman ir raman fully_connected \
  --trials 20 \
  --epochs 20
```

If `--models` is omitted, all four models are included. For each trial, the
minimum validation loss across all epochs is obtained for each model, and their
mean is used as the shared objective value. You can also optimize for a single
model by specifying only that model.

```bash
pipenv run python scripts/optimize_hyperparameters.py \
  --models ir_raman \
  --trials 30 \
  --epochs 20 \
  --output-dir results/tuning_ir_raman
```

Results are saved to `results/hyperparameter_optimization/` by default.

- `best_hyperparameters.json`: Best learning rate, dropout, and objective value
- `trials.csv` / `trials.json`: All trials and per-model validation losses
- `trial_XXXX/<model>/history.json`: Training history for each epoch
- `trial_XXXX/<model>/train.log`: Log for each training process

To avoid generating a large number of files during the search, `best.pt` and
`last.pt` for each trial are deleted by default. Specify `--keep-checkpoints` to
retain them. An interrupted search with the same configuration can be resumed
with `--resume`. After determining the optimal values, train each `train.py`
again with those values and the desired number of production epochs.

If training stops early because the loss or gradient becomes non-finite for a
combination, that trial is recorded as `failed` with the reason in `trials.json`
and `trials.csv`, and the remaining search continues. See the corresponding
trial's `train.log` for details.

To inspect the commands without running any training, use `--dry-run`.

```bash
pipenv run python scripts/optimize_hyperparameters.py --trials 2 --epochs 2 --dry-run
```

## 3. Evaluate a Model

```bash
pipenv run python ir_raman_freq/evaluate.py --split test
```

This reports cross-entropy loss, Top-1, Top-3, and Top-5 structure-match rates
for SMILES canonicalized with RDKit, and the valid-SMILES rate of the Top-1
candidate. Because the Top-3 and Top-5 candidates are generated with beam
search, their evaluation takes longer than Top-1-only evaluation.

Evaluation results are printed to the console and automatically saved as
`evaluation_<split>.json` in the same directory as the checkpoint. For example,
the default path for the test split is
`results/ir_raman_transformer/evaluation_test.json`. To change the output path,
specify `--output results/my_evaluation.json`.

### Final-Model Analysis by Molecular Property and Functional Group

To distinguish final-analysis data from data used for ordinary hyperparameter
searches, specify `--analysis` with `prepare_data.py` only when running the final
analysis. This additionally saves an `analysis_manifest.csv` file containing
the source-pickle position, SMILES, and molecular properties in the same row
order as each split's tensors.

```bash
pipenv run python scripts/prepare_data.py --analysis
pipenv run python ir_raman_freq/evaluate.py --split test --analysis
pipenv run python scripts/analyze_molecular_performance.py \
  --input results/ir_raman_transformer/evaluation_test_molecules.csv
```

The analysis uses `dipole[3]`, `vip[0]`, `vea[0]`, `homolumo[0]`,
`polar_aniso`, `polar_iso`, and `deen`. It also computes RDKit descriptors and
SMARTS-based functional groups, and outputs correlations with numerical
properties, logistic-regression coefficients adjusted for heavy-atom count,
accuracies with and without each functional group and their 95% confidence
intervals, Fisher's exact tests, multiple-testing corrections, and figures.
Figures comparing point-biserial correlations for numerical properties and
accuracy differences with and without functional groups display Top-1, Top-3,
and Top-5 results side by side using a shared color scheme. The corresponding
CSV files include `target` and `top_k` columns that identify the comparison.
`--target` selects the target for summaries and binned accuracies. Figures use
large, publication-ready type and are saved as both 300 dpi PNG files and vector
PDF files. Abbreviations in figures are standardized to forms such as `VIP`,
`VEA`, `TPSA`, `logP`, and `HOMO–LUMO`.

Analysis results are saved by default to
`results/ir_raman_transformer/evaluation_test_molecules_analysis/`. Molecules
are joined using the `size*.pickle` filename and original row number rather than
SMILES. During evaluation, the canonical structure of the manifest SMILES is
also verified against the ground-truth SMILES.

## 4. Identify Molecules from Unknown Spectra

The input CSV must contain the three columns `freq`, `IR`, and `Raman`. Each cell
must be a numeric array of the same length.

```csv
sample_id,freq,IR,Raman
sample-1,"[650.0, 712.3]","[0.2, 1.4]","[3.1, 0.8]"
```

```bash
pipenv run python ir_raman_freq/predict.py --input spectra.csv --top-k 5
```

The top prediction is saved in `predicted_smiles`, while the top candidates and
their mean log probabilities are saved in `candidates_json`. Make sure the
preprocessing of inference inputs—including units, computational conditions,
and intensity scaling—matches the preprocessing used for training.

Run any command with `--help` to see all available options.

## 5. Model-Specific Directories

The three input configurations are organized into separate directories. Run all
commands from the repository root. Each configuration uses the shared
Transformer implementation and training data in `data_diretory/processed/`.
The two-component variants change the encoder input dimension from three to two;
they do not zero-fill an unused spectrum.

| Variant | Command directory | Default model output directory |
|---|---|---|
| Frequency + IR + Raman | `ir_raman_freq/` | `results/ir_raman_transformer/` |
| Frequency + IR | `ir_freq/` | `results/ir_freq_transformer/` |
| Frequency + Raman | `raman_freq/` | `results/raman_freq_transformer/` |

```bash
# Frequency + IR + Raman
pipenv run python ir_raman_freq/train.py --epochs 100 --dropout 0.1
pipenv run python ir_raman_freq/evaluate.py --split test

# Frequency + IR
pipenv run python ir_freq/train.py --epochs 100 --dropout 0.1
pipenv run python ir_freq/evaluate.py --split test

# Frequency + Raman
pipenv run python raman_freq/train.py --epochs 100 --dropout 0.1
pipenv run python raman_freq/evaluate.py --split test
```

Training results, histories, configurations, and prediction CSV files are
written to separate output directories for each model. `scripts/` contains data
preparation tools and internal implementations shared by the models. Legacy
commands such as `python scripts/train.py` remain available for backward
compatibility, but the model-specific entry points above are recommended for
normal use.

## 6. Fully Connected Baseline Model

The non-autoregressive fully connected baseline model is located in
[`fully_connected_model/`](fully_connected_model/). After applying a padding
mask, it concatenates and flattens the three Frequency, IR, and Raman components,
then predicts all SMILES positions simultaneously using a fully connected
encoder and decoder. It does not use attention, RNNs, or CNNs.

It uses the same `data_diretory/processed/` data as the Transformer models. Run
the following commands from the repository root.

```bash
# Train
pipenv run python -m fully_connected_model.train \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --dropout 0.1

# Canonical Top-1, Top-3, and Top-5 evaluation
pipenv run python -m fully_connected_model.evaluate --split test

# Predict from a CSV file
pipenv run python -m fully_connected_model.predict \
  --input spectra.csv \
  --top-k 5
```

The default model output directory is `results/fully_connected_baseline/`. The
inference CSV must contain the three columns `freq`, `IR`, and `Raman`. The main
configurable parameters are listed below. Evaluation results are saved by
default to `results/fully_connected_baseline/evaluation_<split>.json`.

| Argument | Description | Default |
|---|---|---:|
| `--hidden-dimension` | Dimension of fully connected hidden layers | 256 |
| `--latent-dimension` | Dimension of the latent vector | 128 |
| `--encoder-layers` | Number of encoder layers | 3 |
| `--decoder-layers` | Number of decoder layers | 3 |
| `--dropout` | Dropout rate | 0.1 |
| `--learning-rate` | Initial learning rate | 1e-4 |

As with the Transformer models, evaluation reports the Top-1, Top-3, and Top-5
structure-match rates for SMILES canonicalized with RDKit, the Top-1 valid-SMILES
rate, and cross-entropy loss. See the
[fully connected model README](fully_connected_model/README.md) for details on
the model architecture and additional options.
