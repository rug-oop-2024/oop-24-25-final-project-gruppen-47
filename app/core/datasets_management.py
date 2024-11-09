import streamlit as sl
import pandas as pd
from typing import IO

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


def create_dataset(data: IO) -> None:
    automl = AutoMLSystem.get_instance()

    dataframe = pd.read_csv(data)
    dataset = Dataset.from_dataframe(dataframe, data.name, f"{data.name}.csv")

    automl.registry.register(dataset)