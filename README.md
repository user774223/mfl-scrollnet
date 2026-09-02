# MFL-ScrollNet

MFL-ScrollNet is a lightweight recurrent sliding-window model for large-scale panoramic MFL pipeline inspection, preserving cross-window context.


## Project layout

```text
src/mfl_scrollnet/
├── config.py                 Typed configuration and validation
├── data/                     Manifest IO, windowing, datasets, augmentation
├── models/                   Backbone, YOLO head, RoIAlign, GRU, full model
├── training/                 Target assignment, objectives, trainer, checkpoints
├── inference/                Sequential engine and global association
├── metrics/                  AP and macro/micro detection metrics
├── utils/                    Box math, devices, serialization, reproducibility
└── cli.py                    Unified train, infer, and evaluate entry point
configs/default.yaml          Article-aligned hyperparameters
examples/manifest.example.json
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
pytest
```
