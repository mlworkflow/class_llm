from zenml import step
from zenml.client import Client
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
import os
import pickle

@step
def import_artifacts(run_id: str) -> Tuple:
    """Import artifacts from a previous pipeline run."""
    client = Client()
    # Path to an existing folder in your artifact store
    prefix = client.active_stack.artifact_store.path
    pipe_run_path = os.path.join(prefix, "local4colab_pipe_run.pkl")
    pipe_run = pickle.load(open(pipe_run_path, "rb"))

    # Access specific steps from the pipeline run
    pipe_run2.steps['datasets'].outputs['output_3'][0].load()

    # Load the artifacts
    train_dataset = step_outputs["train_dataset"].load()
    val_dataset = step_outputs["val_dataset"].load()
    NUM_LABELS = step_outputs["NUM_LABELS"].load()
    id2label = step_outputs["id2label"].load()
    label2id = step_outputs["label2id"].load()
    weights = step_outputs["weights"].load()

    return train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights