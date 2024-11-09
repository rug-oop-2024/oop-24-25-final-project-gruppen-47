import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
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

