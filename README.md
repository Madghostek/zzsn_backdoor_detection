# ZZSN_backdoor_detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Comparing a few detection methods

## Setup
1. Create environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/acticate # or according to your shell
pip install .
```

2. Setup kaggle if needed, make sure you joined competition. Or download everything by hand
```

Download competition data:
```bash
$ scripts/download.sh
```

3. Download models from https://drive.google.com/drive/folders/1FWc4Krrm1Va6bmPhOjcAp7bPJMQoyr3w into `models/`

## Project Organization

Modified from cookiecuter template

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         zzsn_backdoor_detection and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
│
├── setup.cfg          <- Configuration file for flake8
│
└── zzsn_backdoor_detection   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes zzsn_backdoor_detection a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

