import numpy as np
import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem

from app.core.modelling.datasets import pick_dataset, select_features
from app.core.modelling.pipeline import (
    select_model,
    select_metric,
    select_split,
    write_metric_results,
    write_predictions,
)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.preprocessing import preprocess_features

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    """
    Write a helper text in the app.
    
    Args:
        text (str): The text to be displayed.
    """
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train"
    "a model on a dataset."
)

selected_dataset = pick_dataset()

if selected_dataset:
    input_features, target_feature = select_features(selected_dataset)

    model = select_model(target_feature.type)

    metrics = select_metric(target_feature.type)

    split = select_split()

st.title("Pipeline Configuration")

st.write(f"**Dataset**: {selected_dataset.name}")
st.write(f"**Model**: {model.__name__}")
st.write(f"**Input Features**: {', '.join([x.name for x in input_features])}")
st.write(f"**Target Feature**: {target_feature.name}")
st.write(f"**Split**: {split}%")
st.write(f"**Metrics**: {', '.join([x.__name__ for x in metrics])}")

st.write("## Results")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

try:
    if st.button("Run Pipeline"):
        # Create a pipeline with the selected configuration
        st.session_state.pipeline = Pipeline(
            dataset=selected_dataset,
            model=model(),
            input_features=input_features,
            target_feature=target_feature,
            metrics=list(metric() for metric in metrics),
            split=split / 100,
        )

        results = st.session_state.pipeline.execute()
        labels = results["labels"]
        eval_results = results["metrics_on_evaluation_set"]
        train_results = results["metrics_on_training_set"]
        predictions = results["predictions"]

        if labels is not None:
            st.write(f"Classes: {", ".join(labels)}")  # noqa: E999
            predictions = [labels[prd] for prd in predictions]

        st.write("### Evaluation Set")
        write_metric_results(eval_results)

        st.write("### Training Set")
        write_metric_results(train_results)

        write_predictions(predictions, input_features, selected_dataset, split)
    
except ValueError as e:
    st.write("Error: please fill all the required fields.")
    st.write(e.args[0])

try:
    artifact_name = st.text_input("Enter your pipeline name:")
    artifact_version = st.text_input("Enter your pipeline version:")

    if st.button("Save Pipeline"):
        assert artifact_name != "", st.write("Field Name cannot be empty.")
        assert artifact_version != "", st.write(
            "Field Version cannot be empty."
        )
        new_artifact = st.session_state.pipeline.to_artifact(
            artifact_name, artifact_version
        )
        automl.registry.register(new_artifact)
except AttributeError:
    st.write("Error: please run the pipeline first.")
except AssertionError:
    pass
