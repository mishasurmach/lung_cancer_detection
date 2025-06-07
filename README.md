# Lung Cancer Detection

## Overview

This project implements a deep learning pipeline for detecting lung cancer from histopathological images of lung tissue. It provides:

* **Python** code structured as a package (`lung_cancer_detection/`) with modules for data loading, model definition, training, and inference.
* A **serving** directory with Docker Compose setup for production deployment (backend + frontend).
* **Hydra** for flexible configuration management (configs live under `conf/`).
* **DVC** (Data Version Control) to manage large assets (the dataset and model checkpoints) stored on Google Drive.

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

````

2. **Activate Conda environment**
```bash
conda create -n lung_cancer_detection_env python=3.10 -y
conda activate lung_cancer_detection_env
````

3. **Install Python dependencies via Poetry**

   ```bash
   poetry install        # installs all packages from pyproject.toml
   ```


4. **Install DVC with Google Drive support**

   ```bash
   pip install "dvc[gdrive]"
   dvc install           # initialize DVC in this Git repo
   ```

4. **Pull dataset and existing models**

   You have two options to obtain the data:

   * **Via DVC + Google Drive** (requires private JSON key):

     0. Install DVC with Google Drive support

         ```bash
         pip install "dvc[gdrive]"
         dvc install           # initialize DVC in this Git repo
         ```
     1. Request the `dvc-service.json` key file privately from the project owner.
     2. Place it in `~/.gcp/dvc-service.json` and configure DVC:

        ```bash
        dvc remote modify drive gdrive_use_service_account true
        dvc remote modify drive --local \
            gdrive_service_account_json_file_path ~/.gcp/dvc-service.json
        ```
     3. Pull all artifacts:

        ```bash
        dvc pull      # downloads `lung_image_sets/` and model checkpoints
        ```

   * **Via public GitHub** (no DVC required):

     Check the link: https://github.com/tampapath/lung_colon_image_set

   To fetch only a subset via DVC, specify the DVC file:

   ```bash
   dvc pull lung_image_sets.dvc       # only dataset
   dvc pull models/epoch*.dvc         # only model checkpoints
   ```

---

## Train

To train a new model from scratch or resume training, use the Hydra-enabled entrypoint:

```bash
python lung_cancer_detection.train
```

By default, this will read settings from `conf/train/` and output a checkpoint to `models/`.

To customize parameters via Hydra:

```bash
python -m lung_cancer_detection.train batch_size=16
```

After training completes, track and push the new checkpoint:

```bash
# add the checkpoint under DVC
dvc add models/model.ckpt
git add models/model.ckpt.dvc .gitignore
git commit -m "feat: add trained model checkpoint"
# push checkpoint to remote storage
dvc push
```

---

## Infer

Run inference:

```bash
python -m lung_cancer_detection.infer
```

---

## Serve (Production)

The `serving/` directory contains a Docker-based deployment for both backend and frontend. To launch the server locally:

1. **Install Docker & Docker Compose** (if not already):

   * [Install Docker Desktop](https://www.docker.com/products/docker-desktop)
   * Ensure `docker` and `docker-compose` commands are in your PATH.

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
