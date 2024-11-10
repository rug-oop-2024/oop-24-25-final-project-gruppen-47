import pickle
import numpy as np
import pandas as pd
import streamlit as st

from app.core.deployment.management import (
    convert_df_to_csv,
    select_pipeline,
    write_summary,
    preprocess_input,
)
from app.core.datasets.management import create_dataset
from app.core.modelling.datasets import (
    pick_dataset,
    select_features,
)
from app.core.system import AutoMLSystem


automl = AutoMLSystem.get_instance()

pipeline = select_pipeline()

# Unpack the pipeline
model = pickle.loads(pipeline.data).model
if model.type == "classification":
    labels = pickle.loads(pipeline.data).labels
else:
    labels = None

write_summary(pipeline, model)

predictions_csv = st.file_uploader(
    "**Upload a CSV file for predictions:**", type="csv"
)

try:
    # Create a dataset from the uploaded CSV file
    create_dataset(predictions_csv)
    selected_dataset = pick_dataset()
    input_features = select_features(selected_dataset, False)[0]

    # Get the input variables and predictions to display
    input_variables = preprocess_input(input_features, selected_dataset)
    predictions = model.predict(input_variables)

    # Convert the predictions to labels if the model is classification
    if labels is not None:
        predictions = [labels[prd] for prd in predictions]

    # Create a DataFrame to display the predictions and display it
    predictions_df = pd.DataFrame(
        input_variables, columns=[f.name for f in input_features]
    )
    predictions_df["Predictions"] = predictions
    st.write(predictions_df)

    # Create a csv and download button for the predictions
    csv = convert_df_to_csv(predictions_df)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )

except ValueError as e:
    # st.write("Please upload a CSV file for predictions.")
    st.write(e.args[0])
