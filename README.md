# LasRepair

LasRepair is a research implementation for iterative repair of erroneous
tabular data with a sequence-to-sequence model.
## Repository Structure

```text
LasRepair/
├── src/lasrepair/
│   ├── repair.py                 # Main training and repair entry point
│   ├── dataset.py                # Sequence-to-sequence dataset wrapper
│   ├── confident_learning.py     # Uncertainty calculation
│   ├── utils.py                  # Repair metrics and utility functions
│   ├── evaluation.py             # Result evaluation entry point
│   ├── llm_instruction.py        # Optional LLM causal-matrix utility
│   ├── embedding2graph.py        # Optional embedding graph utility
│   ├── max_modularity.py         # Spectral modularity utilities
│   ├── paths.py                  # Repository-relative default paths
│   └── data/
│       ├── corruption.py
│       └── create_error_versions.py
├── datasets/
├── results/
├── requirements.txt
└── .env.example
```

## Installation

The original project did not specify a Python version. Create an isolated
environment with a Python version supported by the listed PyTorch and
Transformers releases, then install:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

The default model is `google-t5/t5-large`. Transformers may download it on
first use. The default device is `cuda:0`.

## Datasets

Datasets are not distributed in this repository because their source,
license, and redistribution status could not be confirmed from the original
project. Place each authorized dataset in this layout:

```text
datasets/
└── <dataset_name>/
    ├── clean.csv
    ├── dirty.csv
    └── <dataset_name>_<error_percent>_error.csv  # optional
```

The clean and dirty tables must have compatible shapes. See
`datasets/README.md` for the dataset names observed in the original project.

## Environment Variables

The core repair entry point does not require an API key. Optional graph and
causal-matrix utilities use environment variables:

```bash
cp .env.example .env
```

Never commit `.env`. The causal-matrix utility may call a paid external API.

## Training and Repair

Training and repair are performed by the same iterative workflow:

```bash
PYTHONPATH=src python -m lasrepair.repair \
  --experiment flight \
  --gpu cuda:0
```

The default input paths are `datasets/<experiment>/dirty.csv` and
`datasets/<experiment>/clean.csv`. Results are written to `results/`.

An error-rate variant can be selected with the original naming convention:

```bash
PYTHONPATH=src python -m lasrepair.repair \
  --experiment hospital \
  --error_rate 0.1
```

Use `--data_dir` and `--output_dir` to override repository-relative paths.
All original algorithm parameters and defaults remain available through
`python -m lasrepair.repair --help`.

## Evaluation

Evaluate a generated repair result with:

```bash
PYTHONPATH=src python -m lasrepair.evaluation --experiment flight
```

This preserves the original normalized edit-distance calculation. The repair
workflow itself also prints the original project F1 calculation after each
iteration.

## Data Preparation

The original error-rate generation utility is available as:

```bash
PYTHONPATH=src python -m lasrepair.data.create_error_versions \
  --experiment hospital \
  --source_error_rate 52
```

`lasrepair.data.corruption.dataset_corrupter` is also retained as a Python
utility. Its random behavior is unchanged and it does not set a seed.


## License

See [LICENSE](LICENSE).
