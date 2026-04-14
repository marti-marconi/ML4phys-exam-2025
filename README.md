# Galaxy Classification - ML4Physics Exam 2025

This repository contains the implementation for the morphological classification of galaxies using the Galaxy10 SDSS dataset as solution for the ML exam: https://github.com/ML4PhysicsTeachingGenova/exam_2025

---

## Setup and installation
To ensure reproducibility on a remote server or local machine, follow these steps:

### 1. Environment Setup
The project uses a virtual environment to manage dependencies.
```bash
# Create the venv (if not already present)
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Jupyter Kernel Configuration

To run the notebooks in the `notebooks/` directory, you must link the virtual environment to Jupyter:
```bash
pip install ipykernel
python -m ipykernel install --user --name=galaxy_env --display-name "galaxy_env"
```
Then open the notebook and select the proper kernel.

### 3. Training Scripts
For long-running training sessions, use the provided script:

- `train_once.py`: main training logic
- `run_train_once.sh`: shell script to launch the training (useful for nohup or cluster submissions)

---

## Project objectives:

1. Classification: build a 10-label classifier for diffeernt galaxy morphology classes

2. Explainability: implementation of an alorithm to visualize the discriminant features of the galaxy morphologies

3. Anomaly Detection: training on a subset of "normal" classes to identify "unseen" classes as anomalies

---

## Repo structure



