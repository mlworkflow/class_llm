import pandas as pd
from typing import Tuple, Annotated
from zenml import step


@step
def label_transform(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "df"],
    Annotated[dict, "id2label"],
    Annotated[dict, "label2id"],
    Annotated[int, "num_labels"]
]:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    # Load and shuffle data
    df['label'] = df['impact'] + df['urgency']
    df['labels'] = df['label'].apply(lambda x: 0 if x == 0  else ( 0 if x == 1 else ( 0 if x == 2 else  ( 0 if x == 3 else  ( 1 if x == 4 else  (2 if x == 5  else (3 if x == 7 else 3 )) ) )  )  ))
    df = df[['text_clean','labels', 'impact']]

    labels = df['labels'].unique().tolist()
    NUM_LABELS= len(labels)
    id2label={id:label for id,label in enumerate(labels)}

    label2id={label:id for id,label in enumerate(labels)}

    return df, id2label, label2id, NUM_LABELS