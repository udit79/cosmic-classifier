# src/cosmic_classifier/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

from .preprocessing import detect_column_types, build_preprocessor
from .models import build_model_candidates

def parse_args():
    ap = argparse.ArgumentParser(description="Train models on cosmicclassifierTraining.csv")
    ap.add_argument("--data", type=str, required=True, help="Path to CSV file")
    ap.add_argument("--target", type=str, default=None, help="Target column (defaults to last column)")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2, help="Test size for 80/20 split")
    ap.add_argument("--outdir", type=str, default=".")
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "metrics").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    target = args.target or df.columns[-1]
    y = df[target]
    X = df.drop(columns=[target])

    # Detect column types and build preprocessor
    colmap = detect_column_types(df, target)
    preproc = build_preprocessor(colmap)

    # Stratify if classification targets are discrete
    stratify = y if y.nunique() / max(len(y), 1) < 0.5 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify
    )

    candidates = build_model_candidates(args.random_state)
    results: Dict[str, Dict] = {}
    best_name, best_f1 = None, -np.inf

    for name, clf in candidates.items():
        pipe = Pipeline([("pre", preproc), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        report = classification_report(y_test, y_pred)

        results[name] = {"accuracy": acc, "precision_macro": prec, "recall_macro": rec, "f1_macro": f1}
        (outdir / "metrics" / f"{name}_classification_report.txt").write_text(report)

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            joblib.dump(pipe, outdir / "models" / "best_model.joblib")

    # Save summary metrics
    summary = {"best_model": best_name, "metrics": results}
    (outdir / "metrics" / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
