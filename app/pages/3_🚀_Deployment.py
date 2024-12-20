import pickle
import pandas as pd
import streamlit as st

from app.core.deployment.management import (
    convert_df_to_csv,
    select_pipeline,
    write_summary,
    preprocess_input,
)
from app.core.modelling.datasets import select_features
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


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

name_csv = predictions_csv.name if predictions_csv else None
data_csv = pd.read_csv(predictions_csv) if predictions_csv else None

try:
    # Create a dataset from the uploaded CSV file and select features
    selected_dataset = Dataset.from_dataframe(
        data_csv, name_csv, f"{name_csv}.csv"
    )
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

except AttributeError:
    st.toast("Please upload a CSV file for predictions.")

except ValueError as e:
    if e.args[0] == "need at least one array to concatenate":
        st.toast("Please select features from the dataset to predict.")
    else:
        st.write(f"Incorrect number of features: {e.args[0]}")
        st.write(
            "This error may be caused by selecting categorical features on "
            "accident. Categorical features are one-hot encoded and may "
            "result in more features."
        )
