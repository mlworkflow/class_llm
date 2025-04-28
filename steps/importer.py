from zenml import step
from zenml.client import Client
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset

@step
def import_artifacts(run_id: str) -> Tuple[Dataset, Dataset, int, Dict[int, int], Dict[int, int], torch.Tensor]:
    """Import artifacts from a previous pipeline run."""
    client = Client()
    
    # Get the artifacts from the previous run
    pipeline_run = client.get_pipeline_run(run_id)
    
    # Access the outputs from the previous run
    train_dataset = pipeline_run.get_output("train_dataset")
    val_dataset = pipeline_run.get_output("val_dataset")
    NUM_LABELS = pipeline_run.get_output("NUM_LABELS")
    id2label = pipeline_run.get_output("id2label")
    label2id = pipeline_run.get_output("label2id")
    weights = pipeline_run.get_output("weights")
    
    return train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights