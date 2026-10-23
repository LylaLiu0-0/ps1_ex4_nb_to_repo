# Titanic: from notebook to repo

Refactor of the Kaggle *Titanic Data Journey* notebook into a modular Python
project (Fundamentals of Data Science, PS1 Ex4).

## Installation

```bash
conda env create -f environment.yml
conda activate titanic
```

This also installs the `titanic` package in editable mode (`pip install -e .`),
so `from titanic import ...` works from anywhere.

## Usage

Open `notebooks/Titanic Data Journey.ipynb`, select the `titanic` environment as
the kernel, and run all cells.

## Repository structure

```
├── data/                  # train.csv, test.csv, gender_submission.csv
├── notebooks/             # analysis notebook (imports functions from src)
├── src/titanic/
│   ├── config.py          # paths and model parameters
│   ├── data.py            # loading and summary tables
│   ├── features.py        # feature engineering
│   ├── plots.py           # plotting helpers
│   └── model.py           # train / predict / evaluate
├── environment.yml
└── pyproject.toml
```