import numpy as np
import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem

from app.core.modelling.datasets import pick_dataset, select_features
from app.core.modelling.pipeline import (
    select_model,
    select_metric,
    select_split,
)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.preprocessing import preprocess_features

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
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

st.write(f"## Pipeline Configuration")

st.write(f"**Dataset**: {selected_dataset.name}")
st.write(f"**Model**: {model.__name__}")
st.write(f"**Input Features**: {', '.join([x.name for x in input_features])}")
st.write(f"**Target Feature**: {target_feature.name}")
st.write(f"**Split**: {split}%")
st.write(f"**Metrics**: {', '.join([x.__name__ for x in metrics])}")

st.write(f"## Results")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

try:
    if st.button("Run Pipeline"):
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
        if labels is not None:
            st.write(f"Classes: {", ".join(labels)}")  # noqa: E999

        st.write(
            f"**Model Parameters**:{st.session_state.pipeline.model.parameters}"
        )
        eval_results = results["metrics_on_evaluation_set"]
        train_results = results["metrics_on_training_set"]

        eval_results_strings = [
            f"{eval_results[i][0].__class__.__name__}:{eval_results[i][1]}"
            for i in range(len(eval_results))
        ]

        train_results_strings = [
            f"{train_results[i][0].__class__.__name__}: {train_results[i][1]}"
            for i in range(len(train_results))
        ]
        st.write(
            f"**Metrics on evaluation set**:  \n"
            f"{"  \n".join(eval_results_strings)}"
        )
        st.write(
            f"**Metrics on training set**:  \n"
            f"{"  \n".join(train_results_strings)}"
        )

        input_results = preprocess_features(input_features, selected_dataset)
        input_data = [data for (feature_name, data, artifact) in input_results]

        input_test_X = [
            vector[int((split/100) * len(vector)) :]
            for vector in input_data
        ]

        input_X = np.concatenate(input_test_X, axis=1)

        predictions_df = pd.DataFrame(input_X, columns=[f.name for f in input_features])
        predictions_df["Predictions"] = results["predictions"]
        if labels is not None:
            predictions_df["Labels"] = [labels[prd] for prd in results["predictions"]]
        st.write(predictions_df)


except ValueError as e:
    st.write("Error: please fill all the required fields.")
    st.write(e)

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
