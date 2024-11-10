import streamlit as st

from app.core.system import AutoMLSystem
from app.core.datasets.management import create_dataset

automl = AutoMLSystem.get_instance()

uploaded_datasets = st.file_uploader(
    "Upload datasets.", type="csv", accept_multiple_files=True
)

for uploaded_dataset in uploaded_datasets:
    create_dataset(uploaded_dataset)
