from pathlib import Path
import sys
# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from steps._1_data_ingestion import data_ingestion
from steps._2_txt_cleaning import txt_cleaning
from steps._3_label_transform import label_transform
from steps._4_split import split
from steps._5_datasets import datasets
from zenml import Model, pipeline, step
from zenml.client import Client
import pickle
import os


client = Client()

@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="impact_classifier",
    ),
)
def local_pipeline():
    """Define a pipeline that runs all pre-training steps locally."""

    # Data Ingestion Step
    data = data_ingestion('http://prowess.co.ke/all_tickets.csv')

    # Handling Missing Values Step
    data2 = txt_cleaning(data)

    data3, id2label, label2id, NUM_LABELS = label_transform(data2)

    # Data Splitting Step
    X_train, X_val, X_test, y_train, y_val, y_test = split(data3)

    train_dataset, val_dataset, test_dataset, weights = datasets(X_train, X_val, X_test, y_train, y_val, y_test)

    # Return all the needed artifacts for the training step
    return train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights

if __name__ == "__main__":
    # Running the pipeline
    run = local_pipeline()
    
    # Get the run ID for reference in Colab
    print(f"Run ID: {run.id}")
    pipe_run = client.get_pipeline_run(run.id)
    # Path to an existing folder in your artifact store
    prefix = client.active_stack.artifact_store.path
    pipe_run_path = os.path.join(prefix, "local4colab_pipe_run.pkl")
    pickle.dump(pipe_run, open(pipe_run_path, "wb"))