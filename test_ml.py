import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model
import os

# Load the census data
data = pd.read_csv('data/census.csv')

# Define variables
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Split the data into train and test sets
train, test = train_test_split(data, test_size=0.20)

# Process the data using process_data function
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Test 1: Check if ML functions return the expected type of result
def test_ml_function_return_type():
    """
    Test to ensure that the inference function returns a numpy array of predictions.
    """
    model = train_model(X_train, y_train)  # Train the model
    preds = inference(model, X_test)  # Run inference on the test data
    
    assert isinstance(preds, np.ndarray), "The inference function should return a numpy array."


# Test 2: Check if the ML model uses the expected algorithm
def test_ml_model_algorithm():
    """
    Test to ensure the ML model is using the LogisticRegression algorithm.
    """
    model = train_model(X_train, y_train)
    
    assert isinstance(model, LogisticRegression), "The model should be a LogisticRegression."


# Test 3: Check if the compute_model_metrics function returns expected values
def test_computing_metrics():
    """
    Test to check if the compute_model_metrics function returns precision, recall, and F1 score.
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    
    # Check the type of the metrics
    assert isinstance(precision, float), "Precision should be a float."
    assert isinstance(recall, float), "Recall should be a float."
    assert isinstance(f1, float), "F1 score should be a float."
    
    # Optional: Assert that these values are within reasonable ranges
    assert 0 <= precision <= 1, "Precision should be between 0 and 1."
    assert 0 <= recall <= 1, "Recall should be between 0 and 1."
    assert 0 <= f1 <= 1, "F1 score should be between 0 and 1."


# Test 4: Check if training and test datasets have the expected size or data type
def test_dataset_size_and_type():
    """
    Test to ensure the training and test datasets are of correct size and data type.
    """
    # Check the type of X_train and X_test
    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array."
    assert isinstance(X_test, np.ndarray), "X_test should be a numpy array."
    
    # Check the type of y_train and y_test
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array."
    assert isinstance(y_test, np.ndarray), "y_test should be a numpy array."
    
    # Check that train and test sizes are as expected (80-20 split)
    assert len(X_train) == pytest.approx(0.8 * len(data), rel=0.05), "Train set size should be 80% of the data."
    assert len(X_test) == pytest.approx(0.2 * len(data), rel=0.05), "Test set size should be 20% of the data."
