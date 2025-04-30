from zenml import step
from zenml.client import Client
from typing import Tuple, Annotated
import torch
from torch.utils.data import Dataset
import os
import pickle

@step
def import_artifacts() -> Tuple[
    Annotated[Dataset, "train_dataset"],
    Annotated[Dataset, "val_dataset"],
    Annotated[int, "num_labels"],
    Annotated[dict, "id2label"],
    Annotated[dict, "label2id"],
    Annotated[torch.Tensor, "weights"]
]:
    """Import artifacts from a previous pipeline run."""
    client = Client()
    # Path to an existing folder in your artifact store
    prefix = client.active_stack.artifact_store.path
    pipe_run_path = os.path.join(prefix, "local4colab_pipe_run.pkl")
    pipe_run = pickle.load(open(pipe_run_path, "rb"))

    # Load the artifacts
    train_dataset = pipe_run.steps['datasets'].outputs['train_dataset'][0].load()
    val_dataset = pipe_run.steps['datasets'].outputs['val_dataset'][0].load()
    NUM_LABELS = pipe_run.steps['label_transform'].outputs['num_labels'][0].load()
    id2label = pipe_run.steps['label_transform'].outputs['id2label'][0].load()
    label2id = pipe_run.steps['label_transform'].outputs['label2id'][0].load()
    weights = pipe_run.steps['datasets'].outputs['weights'][0].load()

    return train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights