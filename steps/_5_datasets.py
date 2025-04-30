import pandas as pd
from zenml import step
from typing import Tuple, Annotated
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import torch
from zenml.client import Client
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased', max_length=1024)

@step
def datasets(X_train: pd.Series, X_val: pd.Series, X_test: pd.Series, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series)  -> Tuple[
    Annotated[Dataset, "train_dataset"],
    Annotated[Dataset, "val_dataset"],
    Annotated[Dataset, "test_dataset"],
    Annotated[torch.Tensor, "weights"]
    ]:
    """"""
    """
    Creates datasets for training, validation, and testing, keeping TF-IDF sparse.
    """
    train_texts   = list(X_train)
    train_labels  = list(y_train)
    test_texts    = list(X_test)
    test_labels   = list(y_test)
    val_texts     = list(X_val)
    val_labels    = list(y_val)

    # Compute class weights (remains the same)
    class_weights = compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(train_labels),
                                            y = train_labels
                                        )
    print("Class Weights:", class_weights)
    weights = torch.tensor(class_weights, dtype=torch.float)

    # Tokenize text (remains the same)
    # Ensure tokenizer is loaded/defined before this step runs
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Create TF-IDF sparse representations (remains the same)
    tfidf_vectorizer = TfidfVectorizer()
    # Fit on training data only
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    # Transform validation and test data
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # *** OPTIMIZATION: ***
    # Do NOT convert TF-IDF to dense format here.
    # Do NOT add TF-IDF to the 'encodings' dictionaries.

    # Create CustomDataset instances, passing the sparse TF-IDF matrices directly
    train_dataset = CustomDataset(train_encodings, train_labels, X_train_tfidf)
    val_dataset = CustomDataset(val_encodings, val_labels, X_val_tfidf)
    test_dataset = CustomDataset(test_encodings, test_labels, X_test_tfidf)

    # The CustomDataset objects now store the TF-IDF data sparsely.
    # ZenML's default materializer (likely pickle-based for complex objects)
    # should handle saving the CustomDataset instance, including the sparse matrix,
    # much more efficiently than saving the dense version.

    return train_dataset, val_dataset, test_dataset, weights


class CustomDataset(Dataset):
    def __init__(self, encodings, labels=None, tfidf_weights=None):
        """
        Initializes the dataset.

        Args:
            encodings: Dictionary containing tokenizer outputs (e.g., 'input_ids', 'attention_mask').
                       Values are expected to be lists or arrays.
            labels: List or array of labels corresponding to the encodings.
            tfidf_weights: A scipy.sparse matrix (e.g., csr_matrix) containing TF-IDF features.
                           Expected to have the same number of rows as the items in encodings.
        """
        self.encodings = encodings
        self.labels = labels
        # Store the sparse matrix directly
        self.tfidf_weights = tfidf_weights

        # Optional: Add a check for consistent lengths
        num_samples = len(self.encodings['input_ids'])
        if self.labels is not None and len(self.labels) != num_samples:
            raise ValueError("Number of samples in encodings and labels does not match.")
        if self.tfidf_weights is not None and self.tfidf_weights.shape[0] != num_samples:
            raise ValueError("Number of samples in encodings and tfidf_weights does not match.")

    def __getitem__(self, idx):
        # 1. Get tokenizer outputs for the index and convert to tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        # 2. Add label tensor if labels are provided
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])

        # 3. Add TF-IDF weights tensor if weights are provided
        if self.tfidf_weights is not None:
            # Retrieve the specific sparse row for the index
            sparse_row = self.tfidf_weights[idx]
            # Convert *only this row* to a dense numpy array using .toarray()
            # .squeeze() removes the outer dimension (since toarray returns [[...]])
            # Convert the dense numpy array to a PyTorch tensor
            item['tfidf_weights'] = torch.tensor(sparse_row.toarray().squeeze(), dtype=torch.float)
 
        return item

    def __len__(self):
        # The length of the dataset is determined by the number of input sequences
        return len(self.encodings['input_ids'])
    

if __name__ == "__main__":
    # Example usage
    client = Client()
    pipe_run = client.get_pipeline_run('4a3fadfd-ad35-480d-bf4a-8b90ef9f5e1c')
    X_train = pipe_run.steps['split'].outputs['X_train'][0].load()
    X_val = pipe_run.steps['split'].outputs['X_val'][0].load()
    X_test = pipe_run.steps['split'].outputs['X_test'][0].load()
    y_train = pipe_run.steps['split'].outputs['y_train'][0].load()
    y_val = pipe_run.steps['split'].outputs['y_val'][0].load()
    y_test = pipe_run.steps['split'].outputs['y_test'][0].load()
    datasets(X_train, X_val, X_test, y_train, y_val, y_test)