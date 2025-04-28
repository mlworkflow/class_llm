import pandas as pd
from zenml import step
from typing import Tuple
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import logging
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from zenml import ArtifactConfig, step
from zenml.client import Client
from steps._5_datasets import CustomDataset
import mlflow

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)

tokenizer =  AutoTokenizer.from_pretrained('bert-base-uncased', max_length=1024)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def train(train_dataset: CustomDataset, val_dataset: CustomDataset, NUM_LABELS: int, id2label: dict, label2id: dict, weights: torch.Tensor) -> AutoModelForSequenceClassification:
    """"""
    mlflow.pytorch.autolog()
    model_bert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_bert = model_bert.to(device)
    trainer = WeightedLossTrainer(
        model=model_bert,
        weights=weights,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    return model_bert


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weights, reduction='none')
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        if "tfidf_weights" in inputs:
            tfidf_weights = inputs.pop("tfidf_weights")
            tfidf_weights = tfidf_weights.view(-1, 1).expand(-1, self.model.config.num_labels).contiguous()
            loss = loss * tfidf_weights
        return (loss.mean(), outputs) if return_outputs else loss.mean()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1-score': f1, 'precision': precision, 'recall': recall}

training_args = TrainingArguments(
    output_dir='./multi-class-logs',
    do_train=True,
    do_eval=True,
    num_train_epochs=7,
    learning_rate=5e-5,
    torch_compile=False,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=200,
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./multi-class-logs',
    logging_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,
    optim="adamw_torch_fused"
)


