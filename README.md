# itu-sdse-project
## Data Science in Production: MLOps and Software Engineering - Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Description

The goal of the project is to refactor the original notebook-based machine learning prototype into a structured MLOps pipeline.

This project implements a reproducible machine learning training pipeline using GitHub Actions and Dagger.
The pipeline automatically updates the dataset, trains a model, and validates the result on every change to the main branch.


## Project Organization

```
.
├── data
│   └── raw
│       ├── data/                 # raw dataset files
│       └── raw_data.csv.dvc      # DVC-tracked raw dataset
│
├── docs
│   ├── diagrams.excalidraw       # diagrams
│   ├── docs
│   │   └── index.md              # documentation
│   ├── mkdocs.yml                # mkdocs configuration
│   ├── project-architecture.png  # rendered architecture diagram
│   └── README.md                 # documentation-specific README
│
├── go
│   ├── go.mod                    # Go module definition
│   ├── go.sum                    # Go dependencies lockfile
│   └── pipeline.go               # Dagger pipeline definition
│
├── itu_sdse_project
│   ├── __init__.py
│   ├── dataset.py                # dataset loading / preparation
│   ├── features.py               # feature engineering
│   └── modeling
│       ├── __init__.py
│       └── train.py              # model training script
│
├── Makefile                      
├── model
│   └── model.pkl                 # trained model artifact (CI output) / not tracked by git / 
├── pyproject.toml                # project and tooling configuration
├── requirements.txt              # Python dependencies
└── README.md                     
```


## Dagger Workflow

The Dagger workflow is implemented in Go and located in the `go/` folder.
It defines a pipeline for handling:
- Setting up the Python environment
- Installing dependencies
- Runnning the python scripts
- Producing a trained model artifact

**Run locally:**
```bash
cd go
go run .
```

## GitHub Actions

GitHub Actions were used to automate the workflow:
- The Dagger pipeline is executed in CI
- The trained model is uploaded as a GitHub artifact named `model`
- The provided model validation action is used to test the trained model

**Trigger – Runs automatically on:**
- pushes to the `main` branch
- pull requests targeting `main`

**Jobs:**
- train
- validate

