import streamlit as st
from app.core.system import AutoMLSystem


automl = AutoMLSystem.get_instance()

deployed_pipelines = automl.registry.list(type="pipeline")
st.write(deployed_pipelines)