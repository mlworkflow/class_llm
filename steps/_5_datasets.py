import pandas as pd
from zenml import step
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset


tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased', max_length=1024)

@step
def datasets(X_train: pd.Series, X_val: pd.Series, X_test: pd.Series, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Tuple[Dataset, Dataset, Dataset]:
    """"""
    train_texts   =   list(X_train)
    train_labels  =   list(y_train)
    test_texts    =   list(X_test)
    test_labels   =   list(y_test)
    val_texts     =   list(X_val)
    val_labels    =   list(y_val)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Create TF-IDF representations
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Combine BERT embeddings with TF-IDF representations
    train_encodings['tfidf'] = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32).tolist()
    val_encodings['tfidf'] = torch.tensor(X_val_tfidf.toarray(), dtype=torch.float32).tolist()
    test_encodings['tfidf'] = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32).tolist()

    train_dataset = CustomDataset(train_encodings, train_labels, X_train_tfidf)
    val_dataset = CustomDataset(val_encodings, val_labels, X_val_tfidf)
    test_dataset = CustomDataset(test_encodings, test_labels, X_test_tfidf)

    return train_dataset, val_dataset, test_dataset


class CustomDataset(Dataset):
    def __init__(self, encodings, labels=None, tfidf_weights=None):
        self.encodings = encodings
        self.labels = labels
        self.tfidf_weights = tfidf_weights

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        if self.tfidf_weights is not None:
            item['tfidf_weights'] = torch.tensor(self.tfidf_weights[idx].toarray()).float()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])