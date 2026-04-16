# Heart Disease Prediction Notebook

## Project Overview
This notebook implements a complete machine learning pipeline using the `processed.cleveland.csv` dataset to predict heart disease. The workflow includes data loading, cleaning, feature preparation, model training, evaluation, saving, and prediction.

## Dataset and Features
The dataset columns are:
- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`
- `target`

## Data Preprocessing
1. Load the CSV file into a pandas DataFrame.
2. Convert `ca` and `thal` columns to numeric, using `errors='coerce'` to replace invalid values with `NaN`.
3. Convert the target column into a binary label:
   - `1` for heart disease present
   - `0` for no heart disease
4. Optionally inspect missing values and drop rows with missing data if needed.

## Model Training
1. Split the dataset into features `X` and target `y`.
2. Use `train_test_split` with `test_size=0.20` and `random_state=42`.
3. Train a `RandomForestClassifier` with `class_weight='balanced'` to reduce bias from class imbalance.

## Evaluation
- Predict on the test set.
- Compute accuracy using `accuracy_score`.
- Display a classification report with precision, recall, and F1-score.

## Model Persistence
- Save the trained model using `joblib.dump(model, 'model_rf')`.
- Load the saved model with `joblib.load('model_rf')`.

## Prediction Example
- Use a single sample input array to make a prediction.
- Print the prediction result as either "patient" or "healthy".

## Author
Aiham Alqasem
