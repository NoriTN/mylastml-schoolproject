"""Project pipelines."""
from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data_processing import pipeline as dpp
from .pipelines.data_science import pipeline as dsp

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    data_processing_pipeline = dpp.create_pipeline()
    data_science_pipeline = dsp.create_pipeline()
    return {
        "dp": data_processing_pipeline,
        "ds": data_science_pipeline,
        "__default__": data_processing_pipeline + data_science_pipeline,
    }


