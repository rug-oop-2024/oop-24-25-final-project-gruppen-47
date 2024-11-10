import pickle
import streamlit as st

from app.core.deployment.management import select_pipeline, write_summary
from app.core.datasets.management import create_dataset
from app.core.modelling.datasets import (
    pick_dataset,
    select_features,
)
from app.core.system import AutoMLSystem


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

predictions = model.predict(predictions_csv)
st.write(predictions)
st.write("Download the predictions.")
st.download_button(
    label="Download predictions",
    data=predictions,
    file_name="predictions.csv",
    mime="text/csv",
)

