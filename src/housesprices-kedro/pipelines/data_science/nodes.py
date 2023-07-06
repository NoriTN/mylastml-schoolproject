import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

def split_data(data: pd.DataFrame, target:pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    #X = data[parameters["features"]]
    X = data
    y = target["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:

    model_pkl = HistGradientBoostingRegressor()
    model_pkl.fit(X_train, np.log1p(y_train["SalePrice"]))
    return model_pkl


def evaluate_model(
        model_pkl: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.

    Returns:
        Artefacts with the coefficient of determination.
    """
    y_pred = model_pkl.predict(X_test)
    score = r2_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    artefacts = pd.DataFrame({"r2_score": score}, index=[0])
    return artefacts
