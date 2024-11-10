import streamlit as st
from typing import List

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


def pick_dataset() -> Dataset:
    """
    Pick a dataset from the registry.

    Returns:
        Dataset: The selected dataset.
    """
    automl = AutoMLSystem.get_instance()
    datasets = automl.registry.list(type="dataset")

    if not datasets:
        st.write("No datasets found. Please upload a dataset.")
        return None

    selected_dataset = st.selectbox(
        "Please choose a dataset.",
        datasets,
        format_func=lambda dataset: dataset.name,
    )

    return artifact_to_dataset(selected_dataset)


def select_features(dataset: Dataset, both: bool = True) ->  tuple[List[Feature], Feature | None]:
    """
    Select input and target features.

    Args:
        dataset (Dataset): The dataset.

    Returns:
        tuple[List[Feature], Feature]:
            The selected input features and target feature.
    """
    features = detect_feature_types(dataset)

    input_features = select_input_features(features)
    if both:
        target_feature = select_target_feature(features)
    else:
        target_feature = None

    return input_features, target_feature


def select_input_features(features: List[Feature]) -> List[Feature]:
    """
    Select input features.

    Args:
        features (List[Feature]): List of features.

    Returns:
        List[Feature]: List of selected input features.
    """
    return st.multiselect(
        "Please select the input features.",
        features,
        format_func=lambda feature: feature.name,
    )


def select_target_feature(features: List[Feature]) -> Feature:
    """
    Select target feature.

    Args:
        features (List[Feature]): List of features.

    Returns:
        Feature: The selected target feature.
    """
    return st.selectbox(
        "Please select the target feature.",
        features,
        format_func=lambda feature: feature.name,
    )


def artifact_to_dataset(artifact: Artifact) -> Dataset:
    """
    Convert an artifact to a dataset.

    Args:
        artifact (Artifact): The artifact.

    Returns:
        Dataset: The dataset.
    """
    return Dataset(
        name=artifact.name,
        version=artifact.version,
        asset_path=artifact.asset_path,
        tags=artifact.tags,
        metadata=artifact.metadata,
        data=artifact.data,
    )
