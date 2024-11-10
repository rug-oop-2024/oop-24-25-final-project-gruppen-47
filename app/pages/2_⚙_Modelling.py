import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.metric import (
    Accuracy,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    Recall,
    RootMeanSquaredError,
)
from autoop.core.ml.model.classification.k_nearest_neighbors import KNearestNeighbors
from autoop.core.ml.model.classification.random_forest_wrap import RandomForest
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.ridge_wrap import Ridge
from autoop.core.ml.model.regression.lasso_wrap import Lasso
from autoop.core.ml.model.classification.naive_bayes_wrap import NaiveBayesModel
from app.core.modelling.datasets import pick_dataset, select_features
from autoop.core.ml.pipeline import Pipeline

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a model on a dataset."
)

selected_dataset = pick_dataset()

input_features, target_feature = select_features(selected_dataset)


if target_feature.type == "categorical":
    st.write(
        "Based on the selected input features, you have to use classification models."
    )
    model = st.selectbox(
        "Please select a model.",
        [KNearestNeighbors, NaiveBayesModel, RandomForest],
        format_func=lambda model: model.__name__,
    )
    metrics = st.multiselect(
        "Please select the metrics to evaluate the model.",
        [Accuracy, Precision, Recall],
        format_func=lambda metric: metric.__name__,
    )
else:
    model = st.selectbox(
        "Please select a model.",
        [Lasso, Ridge, MultipleLinearRegression],
        format_func=lambda model: model.__name__,
    )
    metrics = st.multiselect(
        "Please select the metrics to evaluate the model.",
        [MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError],
        format_func=lambda metric: metric.__name__,
    )

# TODO change metrics to have attribute categorical or numerical

split = st.slider(
    "Please select the percentage of the dataset that will go for training.", 0, 100, 50
)

st.write(f"## Pipeline Configuration")

st.write(f"**Dataset**: {selected_dataset.name}")
st.write(f"**Model**: {model.__name__}")
st.write(f"**Input Features**: {', '.join([x.name for x in input_features])}")
st.write(f"**Target Feature**: {target_feature.name}")
st.write(f"**Split**: {split}%")
st.write(f"**Metrics**: {', '.join([x.__name__ for x in metrics])}")

st.write(f"## Results")

run = st.button("Run Pipeline")

if run:
    pipeline = Pipeline(
        dataset=selected_dataset,
        model=model(),
        input_features=input_features,
        target_feature=target_feature,
        metrics=list(metric() for metric in metrics),
        split=split / 100,
    )
    results = pipeline.execute()
    st.write(f"**Model Parameters**: {pipeline.model.parameters}")
    st.write(f"**Metrics on evaluation set**: {results["metrics_on_evaluation_set"]}")
    st.write(f"**Metrics on training set**: {results["metrics_on_training_set"]}")
    st.write(f"**Predictions**: {results["predictions"]}")

    # artifact_name = st.text_input("Enter your pipleine name:")
    save = st.button("Save Pipeline")
    print(save)

    if save:
        new_artifact = pipeline.to_artifact("jozo")
        st.write(type(new_artifact))
        print(type(new_artifact))
        automl.registry.register(new_artifact)
        new_artifact.save()
