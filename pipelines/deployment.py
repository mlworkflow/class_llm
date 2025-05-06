import os

from pipelines.training_pipeline import ml_pipeline
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""
    # Run the training pipeline
    trained_model = ml_pipeline()  # No need for is_promoted return value anymore

    # (Re)deploy the trained model
    mlflow_model_deployer_step(workers=1, deploy_decision=True, model=trained_model)
