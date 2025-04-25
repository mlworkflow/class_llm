import pandas as pd
from zenml import step
from typing import Tuple
from sklearn.model_selection import train_test_split


@step
def split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """"""
    X = df['text_clean']
    y = df['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1 , stratify=y, )
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, random_state=1 , stratify=y_train, ) # 0.25 x 0.8 = 0.2

    return X_train, X_val, X_test, y_train, y_val, y_test