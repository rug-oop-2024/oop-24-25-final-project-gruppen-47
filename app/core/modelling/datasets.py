import streamlit as st
import pandas as pd
from typing import IO, List

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types

def pick_dataset() -> Dataset:
    automl = AutoMLSystem.get_instance()
    datasets = automl.registry.list(type="dataset")

    selected_dataset = st.selectbox(
        "Please choose a dataset.",
        datasets,
        format_func=lambda dataset: dataset.name,
        )
    
    return artifact_to_dataset(selected_dataset)

def select_features(dataset: Dataset) -> tuple[List[Feature], Feature]:
    automl = AutoMLSystem.get_instance()
    features = detect_feature_types(dataset)

    input_features = select_input_features(features)
    target_feature = select_target_feature(features)

    return input_features, target_feature

def select_input_features(features: List[Feature]) -> List[Feature]:
    return st.multiselect(
        "Please select the input features.",
        features,
        format_func=lambda feature: feature.name,
    )

def select_target_feature(features: List[Feature]) -> Feature:
    return st.selectbox(
        "Please select the target feature.",
        features,
        format_func=lambda feature: feature.name,
    )
    
def artifact_to_dataset(artifact: Artifact) -> Dataset:
    return Dataset(
        name=artifact.name,
        version=artifact.version,
        asset_path=artifact.asset_path,
        tags=artifact.tags,
        metadata=artifact.metadata,
        data=artifact.data,
    )