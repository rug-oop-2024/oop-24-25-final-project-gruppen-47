from typing import Literal, List
import numpy as np
import pandas as pd
import streamlit as st

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.model import (
    get_model,
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    Model,
)
from autoop.core.ml.metric import (
    get_metric,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    Metric,
)
from autoop.functional.preprocessing import preprocess_features


def select_model(type: Literal["categorical", "numerical"]) -> Model:
    """
    Select a model.

    Args:
        type (Literal["categorical", "numerical"]): Type of the model.

    Returns:
        None
    """
    if type == "categorical":
        st.write(
            "Based on the selected input features, "
            "you have to use classification models."
        )
        model_list = CLASSIFICATION_MODELS
    else:
        st.write(
            "Based on the selected input features, "
            "you have to use regression models."
        )
        model_list = REGRESSION_MODELS

    model = st.selectbox(
        "Please choose a model.",
        model_list,
    )

    return get_model(model)


def select_metric(type: Literal["categorical", "numerical"]) -> List[Metric]:
    """
    Select a metric.

    Args:
        type (Literal["categorical", "numerical"]): Type of the model.

    Returns:
        None
    """
    if type == "categorical":
        metric_list = CLASSIFICATION_METRICS
    else:
        metric_list = REGRESSION_METRICS

    metrics = st.multiselect(
        "Please choose metrics to evaluate the model.",
        metric_list,
    )

    return [get_metric(metric) for metric in metrics]


def select_split() -> int:
    """
    Select the split percentage.

    Returns:
        int: The split percentage.
    """
    split = st.slider(
        "Please select the percentage of the dataset that will go to training",
        0,
        100,
        50,
    )

    return split


def write_metric_results(metric_results: List[tuple[Metric, float]]) -> None:
    """
    Write the metric results.

    Args:
        metric_results (List[tuple[Metric, float]]): The metric results.
    """
    for metric, result in metric_results:
        st.write(f"{metric.__class__.__name__}: {result}")


def write_predictions(
    predictions: List[float],
    features: List[Feature],
    dataset: Dataset,
    split: float,
) -> None:
    """
    Write the predictions.

    Args:
        predictions (List[float]): The predictions.
        features (List[Feature]): The features.
        dataset (Dataset): The dataset.
        split (float): The split percentage.
    """
    st.write("## Predictions")
    input_variables = preprocess_input(features, dataset, split)

    predictions_df = pd.DataFrame(
        input_variables, columns=[f.name for f in features]
    )
    predictions_df["Predictions"] = predictions
    st.write(predictions_df)


def preprocess_input(
    features: List[Feature], dataset: Dataset, split: float
) -> np.ndarray:
    """
    Preprocess the features.

    Args:
        features (List[Feature]): The features.
        dataset (Dataset): The dataset.
        split (float): The split percentage.

    Returns:
        np.ndarray: The preprocessed features.
    """
    input_results = preprocess_features(features, dataset)
    input_data = [data for (feature_name, data, artifact) in input_results]

    input_test_X = [
        vector[int(split * len(vector)) :] for vector in input_data
    ]

    return np.concatenate(input_test_X, axis=1)


def get_name_and_version() -> tuple[str, str]:
    """
    Get the name and version of the pipeline.

    Returns:
        tuple[str, str]: The name and version of the pipeline.
    """
    artifact_name = st.text_input("Enter your pipeline name:")
    artifact_version = st.text_input("Enter your pipeline version:")

    return artifact_name, artifact_version
