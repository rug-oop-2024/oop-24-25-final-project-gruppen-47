from typing import Literal, List
import streamlit as st

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
