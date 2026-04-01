# Customer Churn Prediction with ANN (TensorFlow)

This project predicts whether a bank customer will churn using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

It includes:
- Data preprocessing (categorical encoding + feature scaling)
- ANN model training
- Saved model and preprocessing artifacts for inference
- A separate notebook for prediction/inference

## Project Structure

```text
Customer Churn Prediction ANN/
|-- data/
|   `-- Churn_Modelling.csv
|-- notebook/
|   |-- training.ipynb
|   |-- prediction.ipynb
|   |-- model.h5
|   |-- label_encoder_gender.pkl
|   |-- onehot_encoder_geo.pkl
|   |-- scalar.pkl
|   `-- logs/
|       `-- fits/
|-- requirements.txt
`-- README.md
```

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Open and run:
- `notebook/training.ipynb`

This notebook:
- Loads and preprocesses the churn dataset
- Trains the ANN model
- Saves artifacts used for inference:
	- `notebook/model.h5`
	- `notebook/label_encoder_gender.pkl`
	- `notebook/onehot_encoder_geo.pkl`
	- `notebook/scalar.pkl`

TensorBoard logs are stored under:
- `notebook/logs/fits/`

To view logs:

```bash
tensorboard --logdir notebook/logs/fits
```

## Inference / Prediction

Open and run:
- `notebook/prediction.ipynb`

This notebook loads the trained model and preprocessing artifacts to make churn predictions for new customer inputs.

## Current Scope

This repository currently contains the model development workflow (training + prediction).

Planned next step:
- Build and deploy a Streamlit dashboard for interactive churn prediction.

## Notes

- If any model/artifact file grows large (close to GitHub limits), use Git LFS.
- Keep all preprocessing artifact filenames consistent so inference works correctly.
