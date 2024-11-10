import pickle
import numpy as np
import streamlit as st

from app.core.deployment.management import select_pipeline, write_summary
from app.core.datasets.management import create_dataset
from app.core.modelling.datasets import (
    pick_dataset,
    select_features,
)
from app.core.system import AutoMLSystem
from autoop.functional.preprocessing import preprocess_features


automl = AutoMLSystem.get_instance()

pipeline = select_pipeline()

model = pickle.loads(pipeline.data).model

write_summary(pipeline, model)

predictions_csv = st.file_uploader(
    "**Upload a CSV file for predictions:**", type="csv"
)
create_dataset(predictions_csv)
selected_dataset = pick_dataset()
input_features = select_features(selected_dataset, False)[0]

input_results = preprocess_features(input_features, selected_dataset)
input_data = [data for (feature_name, data, artifact) in input_results]

input_X = np.concatenate(input_data, axis=1)

predictions = model.predict(input_X)
st.write(predictions)
