from zenml import step
from zenml.client import Client
from zenml.models import ArtifactVersionResponse
from typing import Tuple, Annotated
import torch
from torch.utils.data import Dataset
import os
from zenml.logger import get_logger
import pickle

logger = get_logger(__name__)

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
    artifact_path  = client.active_stack.artifact_store.path
    artifact_store_id = client.active_stack.artifact_store.id

    # Load the artifacts
    train_dataset = load_on_colab(pipe_run.steps['datasets'].outputs['train_dataset'][0], 'datasets', artifact_path, artifact_store_id)
    val_dataset = load_on_colab(pipe_run.steps['datasets'].outputs['val_dataset'][0], 'datasets', artifact_path, artifact_store_id)
    NUM_LABELS = load_on_colab(pipe_run.steps['label_transform'].outputs['num_labels'][0], 'label_transform', artifact_path, artifact_store_id)
    id2label = load_on_colab(pipe_run.steps['label_transform'].outputs['id2label'][0], 'label_transform', artifact_path, artifact_store_id)
    label2id = load_on_colab(pipe_run.steps['label_transform'].outputs['label2id'][0], 'label_transform', artifact_path, artifact_store_id)
    
    weights = load_on_colab(pipe_run.steps['datasets'].outputs['weights'][0], 'datasets', artifact_path, artifact_store_id)

    return train_dataset, val_dataset, NUM_LABELS, id2label, label2id, weights
    


def load_on_colab(o: ArtifactVersionResponse, step_name, artifact_path, artifact_store_id):
    o.body.artifact_store_id = artifact_store_id
    normalized_original_uri = o.uri

    # 3. Find the part of the path starting from the step_name directory
    # We look for '/step_name/' to ensure we match the directory, not just a substring
    search_pattern = f"/{step_name}/"
    try:
        # Find the index where '/step_name/' starts (use rindex for potentially nested structures)
        index = normalized_original_uri.rindex(search_pattern)
        # Keep the part starting from '/step_name/' (i.e., including the step name itself)
        relative_path_from_step = normalized_original_uri[index:].lstrip('/') # Remove leading '/' if any
        logger.debug(f"Relative path from step '{step_name}': {relative_path_from_step}")
    except ValueError:
        logger.error(f"Pattern '{search_pattern}' not found in normalized URI: {normalized_original_uri}")
        # Handle error: Maybe the URI format is unexpected, or step_name is wrong
        raise ValueError(f"Step name pattern '/{step_name}/' not found in original URI '{o.uri}'")

    # 4. Construct the new URI by joining the Colab base path and the relative path
    # os.path.join correctly handles joining the base path and the subsequent path part
    new_uri = os.path.join(artifact_path, relative_path_from_step)
    logger.debug(f"Constructed new URI: {new_uri}")

    # 5. Update the URI attribute of the input object
    o.body.uri = new_uri

    return o.load()


if __name__ == "__main__":
    # Running the step
    import_artifacts()