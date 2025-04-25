import pandas as pd
from zenml import step


@step
def data_ingestion(file_path: str) -> pd.DataFrame:
    """Ingest data from a ZIP file using the appropriate DataIngestor."""
    # Load and shuffle data
    df = pd.read_csv(file_path)
    df = df.sample(frac=1).reset_index(drop=True)
    df.head()
    return df