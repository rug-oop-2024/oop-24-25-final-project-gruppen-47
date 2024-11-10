import streamlit as st

from app.core.system import AutoMLSystem

from app.core.modelling.datasets import pick_dataset, select_features
from app.core.modelling.pipeline import select_model, select_metric
from autoop.core.ml.pipeline import Pipeline

automl = AutoMLSystem.get_instance()

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a model on a dataset."
)

selected_dataset = pick_dataset()

input_features, target_feature = select_features(selected_dataset)

model = select_model(target_feature.type)

metrics = select_metric(target_feature.type)

split = st.slider(
    "Please select the percentage of the dataset that will go for training.",
    0,
    100,
    50,
)

st.write(f"## Pipeline Configuration")

st.write(f"**Dataset**: {selected_dataset.name}")
st.write(f"**Model**: {model.__name__}")
st.write(f"**Input Features**: {', '.join([x.name for x in input_features])}")
st.write(f"**Target Feature**: {target_feature.name}")
st.write(f"**Split**: {split}%")
st.write(f"**Metrics**: {', '.join([x.__name__ for x in metrics])}")

st.write(f"## Results")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

try:
    if st.button("Run Pipeline"):
        st.session_state.pipeline = Pipeline(
            dataset=selected_dataset,
            model=model(),
            input_features=input_features,
            target_feature=target_feature,
            metrics=list(metric() for metric in metrics),
            split=split / 100,
        )
        results = st.session_state.pipeline.execute()
        st.write(
            f"**Model Parameters**: {st.session_state.pipeline.model.parameters}"
        )
        st.write(
            f"**Metrics on evaluation set**: {results["metrics_on_evaluation_set"]}"
        )
        st.write(
            f"**Metrics on training set**: {results["metrics_on_training_set"]}"
        )
        st.write(f"**Predictions**: {results["predictions"]}")
except ValueError:
    st.write("Error: please fill all the required fields.")

try:
    artifact_name = st.text_input("Enter your pipleine name:")
    artifact_version = st.text_input("Enter your pipleine version:")

    if st.button("Save Pipeline"):
        assert artifact_name != "", st.write("Field Name cannot be empty.")
        assert artifact_version != "", st.write(
            "Field Version cannot be empty."
        )
        new_artifact = st.session_state.pipeline.to_artifact(
            artifact_name, artifact_version
        )
        automl.registry.register(new_artifact)
except AttributeError:
    st.write("Error: please run the pipeline first.")
except AssertionError:
    pass
