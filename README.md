# CliffordFightNet: Clifford-Inspired Motion Features for Boxing Punch Classification

CliffordFightNet is a PyTorch research project for fine-grained boxing punch classification from short video clips. The goal is to explore whether a Clifford algebra-inspired motion representation can help distinguish punches by combining appearance cues with directional, temporal, and interaction-based motion features.

This repository is being built as a clean, reproducible ML codebase first, with the modeling work added in stages. The current version focuses on project structure, dataset workflow, and engineering foundations rather than a finished benchmark model.

## Motivation

Boxing punch recognition is a strong use case for geometric motion features because punch identity depends on more than static appearance. A jab, cross, hook, or uppercut can differ in motion direction, timing, target zone, and spatial-temporal structure across only a short clip. That makes the task a good fit for feature representations that try to encode how motion components relate to one another, rather than treating each frame independently.

## Technical Honesty

This project is intentionally described as *Clifford-inspired*. The initial representation is not a full mathematical Clifford algebra implementation yet. The first version uses structured multivector-style motion channels and handcrafted interaction terms as a practical stepping stone toward a richer geometric representation.

## Planned Input Features

The planned input pipeline will combine standard visual information with motion-derived channels:

- RGB appearance
- Horizontal gradient `Dx`
- Vertical gradient `Dy`
- Temporal difference `Dt`
- Interaction term `Dx * Dy`
- Interaction term `Dx * Dt`
- Interaction term `Dy * Dt`

The idea is to approximate a multivector-style motion description that captures both primary directional changes and pairwise interactions between spatial and temporal axes.

## Planned Models

- RGB-only 3D CNN baseline
- `CliffordMotionNet` using motion multivector features
- Future true Clifford geometric product block

The near-term plan is to establish a strong baseline first, then measure whether the motion feature design adds meaningful value.

## Dataset

The intended dataset source is the **Olympic Boxing Punch Classification Video Dataset** from Kaggle. The repository does not include downloaded dataset files, and raw data should never be committed to version control.

Important dataset handling rules:

- Keep source videos under `data/raw/`
- Store derived artifacts under `data/processed/`
- Do not commit Kaggle credentials
- Do not commit downloaded clips, processed tensors, or checkpoints

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

### 2. Install requirements

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run the dataset setup script

```bash
python scripts/download_dataset.py
```

At the current stage, this script safely prepares the expected dataset directories and prints guidance rather than performing a full automated download.

### 4. Inspect the dataset directory

```bash
python scripts/inspect_dataset.py
```

### 5. Run tests

```bash
pytest
```

## Repository Structure

```text
clifford-fightnet/
|-- README.md
|-- LICENSE
|-- .gitignore
|-- requirements.txt
|-- pyproject.toml
|-- configs/
|   `-- default.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- notebooks/
|   `-- README.md
|-- scripts/
|   |-- download_dataset.py
|   `-- inspect_dataset.py
|-- src/
|   `-- clifford_fightnet/
|       |-- __init__.py
|       |-- config.py
|       |-- constants.py
|       |-- data/
|       |   |-- __init__.py
|       |   `-- dataset.py
|       |-- models/
|       |   |-- __init__.py
|       |   |-- baseline_3dcnn.py
|       |   `-- clifford_motion.py
|       |-- training/
|       |   |-- __init__.py
|       |   |-- train.py
|       |   `-- evaluate.py
|       `-- utils/
|           |-- __init__.py
|           |-- reproducibility.py
|           `-- paths.py
|-- tests/
|   `-- test_imports.py
`-- outputs/
    `-- .gitkeep
```

## Planned Experiments

The initial experiment plan is straightforward and measurable:

- Baseline vs `CliffordMotionNet`
- Accuracy
- Macro F1
- Confusion matrix
- Class-wise analysis

The purpose is not just to report a single score, but to understand where motion-aware features help and where they fail.

## Roadmap

- [x] Repo setup
- [ ] Dataset loading
- [ ] Baseline model
- [ ] Clifford-inspired feature extractor
- [ ] Training pipeline
- [ ] Evaluation
- [ ] Visualization
- [ ] Colab notebook
- [ ] True Clifford product block

## Limitations

- This is not a state-of-the-art system yet.
- Results will depend heavily on dataset quality, class balance, and annotation consistency.
- The initial version uses handcrafted geometric interaction terms rather than a full Clifford algebra formulation.

## Why This Project Matters

This project matters because it sits at the intersection of deep learning, geometric reasoning, and applied sports video understanding. For recruiters and collaborators, it demonstrates more than model training: it shows the ability to frame a research hypothesis, build a disciplined ML repository, reason honestly about mathematical inspiration versus implementation reality, and design experiments that test whether a new representation is actually useful.

In practical terms, CliffordFightNet is an example of taking an abstract idea from geometry-inspired representation learning and turning it into a testable computer vision project with reproducible engineering practices.

## Current Status

The repository currently provides:

- A professional `src/`-based package structure
- Config-driven paths and hyperparameters
- Reproducibility utilities
- Safe dataset scripts
- Placeholder dataset, model, and training modules
- Basic tests for importability and project health

The full model and training pipeline are still to be implemented.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
