# pip install pandas numpy scikit-learn liac-arff
import numpy as np
import pandas as pd
import arff  

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_nasa93_arff(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = arff.load(f)

    cols = [c[0] for c in data["attributes"]]
    df = pd.DataFrame(data["data"], columns=cols)

    df = df.replace("?", pd.NA)

    for c in ["year", "equivphyskloc", "act_effort"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def filter_missing_and_split(df: pd.DataFrame):

    df = df.dropna(subset=["act_effort"]).copy()

    drop_cols = [c for c in ["recordnumber", "projectname"] if c in df.columns]
    y = df["act_effort"]
    X = df.drop(columns=["act_effort"] + drop_cols, errors="ignore")

    return X, y


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clipping por cuantiles (winsorizing) para reducir el impacto de outliers."""
    def __init__(self, low=0.01, high=0.99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.low, axis=0)
        self.hi_ = np.nanquantile(X, self.high, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.columns.difference(cat_cols)

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper(low=0.01, high=0.99)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre


if __name__ == "__main__":
    df = load_nasa93_arff("data/nasa93/nasa93.arff")
    print("Shape raw:", df.shape)
    print("Missing per column (top):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    X, y = filter_missing_and_split(df)
    preprocessor = build_preprocessor(X)

    print("X shape:", X.shape, "y shape:", y.shape)
    print("Prepared preprocessor OK")