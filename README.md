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

## Project objectives

1. Build a ten-label classifier able to distinguish the galaxies, and evaluate its performance using the most appropriate evaluation metrics
2. By considering the classifier developed in 1. implement a strategy for inspecting the content of the image for understanding the portions with more discriminant information. In particular, choose one explainability method (e.g., saliency maps, feature map visualization, Grad-CAM, or another interpretable algorithm), and use it to analyze how the model arrived at its decisions
3. Build an algorithm to efficiently perform an anomaly-detection task by considering some of the classes for training and treating the others as unseen.

---

## Repo structure

```text
.
├── artifacts/       # Saved models (.keras) and training metrics (JSON)
├── data/            # Galaxy10.h5 dataset (not tracked by git)
├── logs/            # Training logs with timestamps (not very interesting but ok)
├── notebooks/       # Jupyter notebooks
├── src/             # Python scripts and shell utilities
├── README.md        # Project documentation
└── requirements.txt # Python dependencies
```

The real interesting thing is the notebook `notebooks/classifier.ipynb`. The scripts in `\src` were usefull to me to better organize myself, accelerate the workflow, and solve some computing issues.

---

## Author 

Martina Marconi (marti-marconi)