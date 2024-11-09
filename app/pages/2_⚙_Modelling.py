from sklearn.linear_model import Lasso, LogisticRegression, Ridge
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import Accuracy, MeanAbsoluteError, MeanSquaredError, Precision, Recall, RootMeanSquaredError
from autoop.core.ml.model.classification.k_nearest_neighbors import KNearestNeighbors
from autoop.core.ml.model.classification.random_forest_wrap import RandomForest
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a model on a dataset."
)

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

selected_dataset = st.selectbox(
    "Please choose a dataset.",
    datasets,
    format_func=lambda dataset: dataset.name,
    )

print(type(selected_dataset))

features = detect_feature_types(selected_dataset)

input_features = st.multiselect("Please select the input features.", features, format_func=lambda feature: feature.name)


if input_features.type == "categorical":
    st.write("Based on the selected input features, you have to use classification models.")
    model = st.selectbox("Please select a model.", [KNearestNeighbors, LogisticRegression, RandomForest])
    metrics = st.multiselect("Please select the metrics to evaluate the model.", [Accuracy, Precision, Recall])
else:
    model = st.selectbox("Please select a model.", [Lasso, Ridge, MultipleLinearRegression])
    metrics = st.multiselect("Please select the metrics to evaluate the model.", [MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError])

split = st.slider("Please select the percentage of the dataset that will go for training.", 0, 100, 50)

