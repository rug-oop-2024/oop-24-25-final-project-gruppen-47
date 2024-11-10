import pickle
import streamlit as st
from app.core.datasets.management import create_dataset
from app.core.modelling.datasets import (
    artifact_to_dataset,
    pick_dataset,
    select_features,
)
from app.core.system import AutoMLSystem


automl = AutoMLSystem.get_instance()

deployed_pipelines = automl.registry.list(type="pipeline")
st.write("**Deployed pipelines:**")
st.write(", ".join(x.name for x in deployed_pipelines))

summary_pipeline = st.selectbox(
    "Select a pipeline to view the summary.",
    deployed_pipelines,
    format_func=lambda x: x.name,
)

model = pickle.loads(summary_pipeline.data).model
st.write(f"**Summary of {summary_pipeline.name}:**")
st.write(f"Version: {summary_pipeline.version}")
st.write(f"type: {summary_pipeline.type}")
st.write(f"Model: {model.__class__.__name__}")

predictions_csv = st.file_uploader(
    "**Upload a CSV file for predictions:**", type="csv"
)
create_dataset(predictions_csv)
selected_dataset = pick_dataset()
input_features, target_feature = select_features(selected_dataset)

predictions = model.predict(predictions_csv)
st.write(predictions)
st.write("Download the predictions.")
st.download_button(
    label="Download predictions",
    data=predictions,
    file_name="predictions.csv",
    mime="text/csv",
)
st.write("Deploy the pipeline.")
if st.button("Deploy"):
    automl.deploy(summary_pipeline)
    st.write("Pipeline deployed successfully.")
    st.write(
        "Refresh the page to view the updated list of deployed pipelines."
    )
    st.balloons()
