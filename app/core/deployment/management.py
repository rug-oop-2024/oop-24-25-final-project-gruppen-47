import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model


def select_pipeline() -> Artifact:
    """
    List the deployed pipelines.

    Returns:
        Artifact: The selected pipeline.
    """
    automl = AutoMLSystem.get_instance()

    deployed_pipelines = automl.registry.list(type="pipeline")
    st.write("**Deployed pipelines:**")
    st.write(", ".join(x.name for x in deployed_pipelines))

    selected_pipeline = st.selectbox(
        "Select a pipeline to view the summary.",
        deployed_pipelines,
        format_func=lambda x: x.name,
    )

    return selected_pipeline


def write_summary(pipeline: Artifact, model: Model) -> None:
    """
    Write the summary of the pipeline.

    Args:
        pipeline (Artifact): The pipeline.
        model (Model): The model.
    """
    st.write(f"**Summary of {pipeline.name}:**")
    st.write(f"Version: {pipeline.version}")
    st.write(f"type: {pipeline.type}")
    st.write(f"Model: {model.__class__.__name__}")
