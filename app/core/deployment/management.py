from typing import List
import numpy as np
import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.model import Model
from autoop.functional.preprocessing import preprocess_features


def select_pipeline() -> Artifact:
    """
    List the deployed pipelines.

    Returns:
        Artifact: The selected pipeline.
    """
    automl = AutoMLSystem.get_instance()

    deployed_pipelines = automl.registry.list(type="pipeline")
    st.write("**Deployed pipelines:**")
    st.write(", ".join(x.name for x in deployed_pipelines))

    selected_pipeline = st.selectbox(
        "Select a pipeline to view the summary.",
        deployed_pipelines,
        format_func=lambda x: x.name,
    )

    return selected_pipeline


def write_summary(pipeline: Artifact, model: Model) -> None:
    """
    Write the summary of the pipeline.

    Args:
        pipeline (Artifact): The pipeline.
        model (Model): The model.
    """
    st.write(f"**Summary of {pipeline.name}:**")
    st.write(f"Version: {pipeline.version}")
    st.write(f"type: {pipeline.type}")
    st.write(f"Model: {model.__class__.__name__}")


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame.

    Returns:
        bytes: The CSV file.
    """
    return df.to_csv(index=False).encode("utf-8")


def preprocess_input(features: List[Feature], dataset: Dataset) -> np.ndarray:
    """
    Preprocess the input data.

    Args:
        features (List[Feature]): The input features.
        dataset (Dataset): The dataset.

    Returns:
        np.ndarray: The preprocessed input data.
    """
    input_results = preprocess_features(features, dataset)
    input_data = [data for (feature_name, data, artifact) in input_results]

    return np.concatenate(input_data, axis=1)
