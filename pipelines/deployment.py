from pathlib import Path
import sys
# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os

from pipelines.training_short4colab import colab_pipeline
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.client import Client

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""
    # client = Client()
    # pipe_run = client.get_pipeline_run(run.id)
    # Run the training pipeline
    trained_model = colab_pipeline()  # No need for is_promoted return value anymore

    # (Re)deploy the trained model
    mlflow_model_deployer_step(workers=1, deploy_decision=True, model=trained_model)


if __name__ == "__main__":
    # Running the pipeline
    run = continuous_deployment_pipeline()
    print(f"Pipeline run started: {run.id}")
