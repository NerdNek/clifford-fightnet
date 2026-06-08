# clifford-fightnet

`clifford-fightnet` is a PyTorch research repository for boxing punch classification from short video clips. The long-term goal is to explore whether a Clifford algebra-inspired motion representation can improve action recognition by combining RGB appearance, spatial gradients, temporal differences, and bivector-style interaction features.

This repository is intentionally starting with a strong engineering foundation before the full model is implemented. That gives us a reproducible, readable, and extensible codebase for experiments as the research direction matures.

## Motivation

Boxing punch recognition sits in an interesting middle ground between fine-grained action recognition and motion understanding. Small differences in guard position, acceleration, and limb interaction can separate a jab from a cross or hook. A Clifford-inspired representation is appealing because it gives us a structured way to reason about motion interactions rather than treating each cue independently.

The planned representation will eventually combine:

- RGB frames for appearance and context
- Spatial gradients for edge and local shape information
- Temporal differences for frame-to-frame motion
- Bivector-style interaction features for directional motion relationships

## Project Status

Current status: repository scaffold and engineering foundation.

What is included today:

- `src/`-based Python package layout
- Config-driven paths and hyperparameters
- Deterministic seed helper
- Safe dataset utility scripts that do not require a downloaded dataset
- Placeholder dataset, model, and training modules
- Lightweight tests for package importability and config loading

What is intentionally not implemented yet:

- Final dataset ingestion pipeline
- Real video preprocessing
- 3D CNN baseline logic
- Clifford-inspired motion encoder
- End-to-end training and evaluation loops

## Planned Architecture

The initial research plan is to build the project in stages:

1. Establish a reproducible training pipeline and dataset interface.
2. Implement a simple RGB-based baseline such as a 3D CNN.
3. Add motion-focused feature channels:
   - spatial gradients
   - temporal differences
   - interaction features inspired by Clifford algebra
4. Compare baselines and enhanced motion representations on boxing punch classes.
5. Analyze error modes, motion confusion patterns, and ablations.

## Repository Layout

```text
clifford-fightnet/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── README.md
├── scripts/
│   ├── download_dataset.py
│   └── inspect_dataset.py
├── src/
│   └── clifford_fightnet/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baseline_3dcnn.py
│       │   └── clifford_motion.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   └── evaluate.py
│       └── utils/
│           ├── __init__.py
│           ├── reproducibility.py
│           └── paths.py
├── tests/
│   └── test_imports.py
└── outputs/
    └── .gitkeep
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the package in editable mode

```bash
pip install -e .
```

### 4. Run the tests

```bash
pytest
```

## Configuration

The default configuration lives in [`configs/default.yaml`](configs/default.yaml). It contains:

- directory paths
- random seed
- dataset split names
- model placeholder settings
- training hyperparameters
- logging defaults

Configuration is loaded through `clifford_fightnet.config.load_config()`, which resolves project-relative paths with `pathlib`.

## Data Handling

This repository keeps a strict separation between raw and processed data:

- `data/raw/`: downloaded or externally provided source data
- `data/processed/`: derived assets created by preprocessing jobs
- `outputs/`: experiment outputs such as logs, metrics, plots, and temporary artifacts

Important rules:

- Do not commit datasets.
- Do not commit Kaggle credentials or any other secrets.
- Do not commit model checkpoints.
- Prefer preprocessing scripts that are deterministic and logged.

The provided scripts are safe placeholders:

- `scripts/download_dataset.py` creates the expected directories and prints guidance for manual dataset setup.
- `scripts/inspect_dataset.py` summarizes file counts without assuming a specific dataset format.

## Development Notes

- Use `pathlib.Path` instead of raw string paths.
- Keep modules small and single-purpose.
- Add concise comments only when they clarify non-obvious behavior.
- Preserve reproducibility by routing parameters through config files and the seed helper.

## Roadmap

- [x] Create clean project scaffold
- [x] Add packaging, config loading, and reproducibility utilities
- [x] Add safe dataset scripts and placeholder modules
- [ ] Define boxing class taxonomy and dataset manifest format
- [ ] Implement raw clip indexing and metadata parsing
- [ ] Build RGB baseline dataloader
- [ ] Implement baseline 3D CNN
- [ ] Add spatial gradient and temporal difference channels
- [ ] Design Clifford-inspired interaction feature module
- [ ] Add training loop, evaluation metrics, and experiment tracking
- [ ] Run ablation studies and benchmark results

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
