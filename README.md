# RNA 3D Folding Project

This project predicts 3D coordinates of RNA sequences using a CNN model, designed for a Kaggle-like competition.

## Directory Structure

- `data/raw/`          : Kaggle input files (e.g., `train_sequences.csv`, `test_sequences.csv`).
- `data/processed/`    : (Optional) Preprocessed data.
- `src/`              : Source code.
  - `data_processing.py` : Data loading and preprocessing.
  - `model.py`          : Model training and saving.
  - `predict.py`        : Prediction and submission generation.
  - `utils.py`          : Helper functions (currently empty).
- `models/`           : Trained model (`rna_model.h5`).
- `app/`              : (Not implemented) API deployment.

## Setup

1. **Place Data**: Copy your Kaggle CSV files into `data/raw/`.
2. **Install Dependencies**: Run `pip install -r requirements.txt`.
3. **Train Model**: Run `python src/model.py` to train and save the model.
4. **Generate Predictions**: Run `python src/predict.py` to create `submission.csv`.

## Notes

- Ensure all data files are in `data/raw/` with the expected names.
- The trained model is saved as `models/rna_model.h5`.