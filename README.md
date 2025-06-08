# Lung Cancer Detection

## Overview

This project implements a deep learning pipeline for detecting lung cancer from
histopathological images of lung tissue. It provides:

- **Python** code structured as a package (`lung_cancer_detection/`) with
  modules for data loading, model definition, training, and inference.
- A **serving** directory with Docker Compose setup for production deployment
  (backend + frontend).
- **Hydra** for flexible configuration management (configs live under `conf/`).
- **DVC** (Data Version Control) to manage large assets (the dataset and model
  checkpoints) stored on Google Drive.

Repository structure:

```
├── conf/                          # Hydra config files
├── lung_cancer_detection/         # Python package (source code)
│   ├── __init__.py
│   ├── dataset.py                 # DataModule: loading & transforms
│   ├── model.py                   # LightningModule: network definition
│   ├── train.py                   # Training entrypoint (Hydra-enabled)
│   └── infer.py                   # Inference entrypoint
├── models/                        # DVC-tracked model checkpoints
│   └── epoch=04-val_loss=0.1926.ckpt.dvc  # pointer file for a checkpoint
├── serving/                       # Docker Compose for production serving
│   ├── backend/                   # Backend server (MLflow)
│   └── frontend/                  # Frontend server (Flask)
├── tests/                         # pytest test suite
├── lung_image_sets.dvc            # DVC pointer to the dataset archive/folder
└── poetry.lock / pyproject.toml   # Poetry-managed dependencies
```

---

## Setup

1. **Clone the repo**

   ```bash
   git clone https://github.com/mishasurmach/lung_cancer_detection.git
   cd lung_cancer_detection
   ```

2. **Activate Conda environment**

   ```bash
   conda create -n lung_cancer_detection_env python=3.10 -y
   conda activate lung_cancer_detection_env
   ```

3. **Install Python dependencies via Poetry**

   ```bash
   poetry install
   ```

4. **Pull dataset and existing models**

   You have two options to obtain the data:

   - **Via DVC + Google Drive** (requires private JSON key):

     1. Install DVC with Google Drive support

        ```bash
        poetry run dvc install 
        ```

     2. Request the `dvc-service.json` key file privately from the project
        owner.
     3. Place it in `~/.gcp/dvc-service.json` and configure DVC:

        ```bash
        dvc remote modify drive gdrive_use_service_account true
        dvc remote modify drive --local \
            gdrive_service_account_json_file_path ~/.gcp/dvc-service.json
        ```

     4. Pull all artifacts:

        ```bash
        poetry run dvc pull
        ```

     To fetch only a subset via DVC, specify the DVC file:
  
       ```bash
       poetry run dvc pull lung_image_sets.dvc       # only dataset
       poetry run dvc pull models/epoch*.dvc         # only model checkpoints
       ```


   - **Via public GitHub** (no DVC required):

     Check the link for the dataset: https://github.com/tampapath/lung_colon_image_set
     Ask the owner for model weights.
---

## Git hooks

To enable git hooks:

```bash
pre-commit install
```

---

## Train
To track your training experiments, in another terminal run

```bash
mlflow server --host 127.0.0.1 --port 8080
```

and go to http://127.0.0.1:8080 .

To train a new model from scratch or resume training, use the Hydra-enabled
entrypoint: in the first terminal

```bash
cd lung_cancer_detection 
python train.py
```

By default, this will read settings from `conf/train/` and output a checkpoint
to `models/`.

To customize parameters via Hydra:

```bash
python -m lung_cancer_detection.train batch_size=16
```
---

## Infer

Run inference: from `lung_cancer_detection/` run

```bash
python infer.py
```

---

## Serve (Production)

The `serving/` directory contains a Docker-based deployment for both backend and
frontend. To launch the server locally:

1. **Install Docker & Docker Compose** (if not already):

   - [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Ensure `docker` and `docker-compose` commands are in your PATH.

2. **Navigate to `serving/` and start the services**:

   ```bash
   cd serving
   docker-compose up -d
   ```

3. **Stop the services**:

   ```bash
   docker-compose down
   ```

---

Happy developing! 🚀
