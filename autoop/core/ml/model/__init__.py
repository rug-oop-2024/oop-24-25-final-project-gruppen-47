"""Module to manage models.

This module contains the factory function to get a model by name.
"""

from autoop.core.ml.model.classification import (
    KNearestNeighbors,
    NaiveBayesModel,
    RandomForest,
)
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (
    MultipleLinearRegression,
    Ridge,
    Lasso,
)

REGRESSION_MODELS = [
    "K Nearest Neighbors",
    "Naive Bayes Model",
    "Random Forest",
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "Lasso",
    "Ridge",
    "Multiple Linear Regression",
]  # add your models as str here


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    model_mapping = {
        "K Nearest Neighbors": KNearestNeighbors,
        "Naive Bayes Model": NaiveBayesModel,
        "Random Forest": RandomForest,
        "Lasso": Lasso,
        "Ridge": Ridge,
        "Multiple Linear Regression": MultipleLinearRegression,
    }
    try:
        return model_mapping[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} not found")
