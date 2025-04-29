import pandas as pd
from zenml import step
from typing import Tuple, Annotated
from sklearn.model_selection import train_test_split


@step
def split(df: pd.DataFrame) -> Tuple[
    Annotated[pd.Series, "X_train"],
    Annotated[pd.Series, "X_val"],
    Annotated[pd.Series, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_val"],
    Annotated[pd.Series, "y_test"]
]:
    """"""
    X = df['text_clean']
    y = df['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1 , stratify=y, )
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=1 , stratify=y_train, ) # 0.25 x 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test