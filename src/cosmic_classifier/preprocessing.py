# src/cosmic_classifier/preprocessing.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def detect_column_types(df: pd.DataFrame, target: str) -> Dict[str, List[str]]:
    cols = [c for c in df.columns if c != target]
    num_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    obj_cols = [c for c in df[cols].select_dtypes(include=["object", "category"]).columns.tolist()]
    # Heuristic: long strings (avg length > 20 or avg tokens > 3) are text; others are categorical
    text_cols, cat_cols = [], []
    for c in obj_cols:
        s = df[c].astype(str).fillna("")
        avg_len = s.str.len().mean()
        avg_tokens = s.str.split().map(len).mean()
        (text_cols if (avg_len > 20 or avg_tokens > 3) else cat_cols).append(c)
    return {"numeric": num_cols, "categorical": cat_cols, "text": text_cols}

def build_preprocessor(colmap: Dict[str, List[str]]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    # Join multiple text columns per row, then vectorize
    text_join = FunctionTransformer(lambda X: X.fillna("").agg(" ".join, axis=1), validate=False)
    text_pipe = Pipeline([
        ("join", text_join),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000))
    ])
    transformers = []
    if colmap["numeric"]:
        transformers.append(("num", numeric_pipe, colmap["numeric"]))
    if colmap["categorical"]:
        transformers.append(("cat", categorical_pipe, colmap["categorical"]))
    if colmap["text"]:
        transformers.append(("txt", text_pipe, colmap["text"]))
    return ColumnTransformer(transformers=transformers, remainder="drop")
