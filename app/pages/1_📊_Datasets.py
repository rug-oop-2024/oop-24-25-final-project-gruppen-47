import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from app.core.datasets_management import create_dataset

automl = AutoMLSystem.get_instance()

uploaded_datasets = st.file_uploader(
    "Upload datasets.", type="csv", accept_multiple_files=True
)

for uploaded_dataset in uploaded_datasets:
    create_dataset(uploaded_dataset)