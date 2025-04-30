from pathlib import Path
import sys
# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from steps.importer import import_artifacts
from steps._6_train import train
from zenml import Model, pipeline
from zenml.models import PipelineRunResponse



@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="impact_classifier",
    ),
)
def colab_pipeline(artifact_path, artifact_store_id):
    """Pipeline that imports artifacts and runs training in Colab."""
    
    # Import artifacts from the local pipeline run
    train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights = import_artifacts(artifact_path, artifact_store_id)
    
    # Train the model
    model_trained = train(train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights)
    
    return model_trained


if __name__ == "__main__":
    # Running the pipeline
    run = colab_pipeline()
    
    # Get the run ID for reference in Colab
    print(f"Run ID: {run.id}")



