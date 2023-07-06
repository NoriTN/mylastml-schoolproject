from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_model, split_data, train_model, gen_artifacts


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["house_ohe_prices_for_model", "house_prices_target", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="model_pkl",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["model_pkl", "X_test", "y_test"],
                outputs="artefacts",
                name="evaluate_model_node",
            ),
            node(
                func=gen_artifacts,
                inputs=["model_pkl", "X_test", "y_test"],
                outputs="my_plot_png",
                name="gen_artifacts_node",
            )
        ]
    )