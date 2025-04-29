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
def colab_pipeline():
    """Pipeline that imports artifacts and runs training in Colab."""
    
    # Import artifacts from the local pipeline run
    train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights = import_artifacts()
    
    # Train the model
    model_trained = train(train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights)
    
    return model_trained



