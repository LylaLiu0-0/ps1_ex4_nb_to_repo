
"""Functions for training and evaluating the Titanic prediction model."""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from titanic.config import PREDICTORS, RANDOM_STATE, RF_PARAMS, TARGET, VALID_SIZE


def split_data(
    df: pd.DataFrame,
    valid_size: float = VALID_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and validation sets."""
    return train_test_split(df, test_size=valid_size, random_state=random_state, shuffle=True)


def get_X_y(
    df: pd.DataFrame,
    predictors: list[str] = PREDICTORS,
    target: str = TARGET,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Separate the features and the target variable."""
    return df[predictors], df[target].values


def train_model(X: pd.DataFrame, y: np.ndarray, params: dict = RF_PARAMS) -> RandomForestClassifier:
    """Train a random forest model using the given data."""
    clf = RandomForestClassifier(**params)
    clf.fit(X, y)
    return clf


def train_and_predict(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    predictors: list[str] = PREDICTORS,
    target: str = TARGET,
    params: dict = RF_PARAMS,
) -> dict:
    """Train the model and make predictions for both datasets."""
    train_X, train_y = get_X_y(train, predictors, target)
    valid_X, valid_y = get_X_y(valid, predictors, target)
    model = train_model(train_X, train_y, params)
    return {
        "model": model,
        "train_y": train_y,
        "train_pred": model.predict(train_X),
        "valid_y": valid_y,
        "valid_pred": model.predict(valid_X),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Evaluate the predictions using precision, recall and F1 score."""
    return metrics.classification_report(
        y_true, y_pred, target_names=["Not Survived", "Survived"]
    )
