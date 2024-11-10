import pickle
import numpy as np
import pandas as pd
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

if model.type == "classification":
    labels = pickle.loads(pipeline.data).labels
else:
    labels = None

write_summary(pipeline, model)

predictions_csv = st.file_uploader(
    "**Upload a CSV file for predictions:**", type="csv"
)
try:
    create_dataset(predictions_csv)
    selected_dataset = pick_dataset()
    input_features = select_features(selected_dataset, False)[0]

    input_results = preprocess_features(input_features, selected_dataset)
    input_data = [data for (feature_name, data, artifact) in input_results]

    input_X = np.concatenate(input_data, axis=1)

    predictions = model.predict(input_X)

    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    if st.button("Show predictions"):
        predictions_df = pd.DataFrame(input_X, columns=[f.name for f in input_features])
        predictions_df["Predictions"] = predictions
        if labels is not None:
            predictions_df["Labels"] = [labels[prd] for prd in predictions]
        csv = convert_df_to_csv(predictions_df)
        st.write(predictions_df)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )
        
except ValueError as e:
    # st.write("Please upload a CSV file for predictions.")
    st.write(e.args[0])
except KeyboardInterrupt:
    pass
