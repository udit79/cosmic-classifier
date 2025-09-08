# src/cosmic_classifier/evaluate.py
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a saved model on a CSV")
    ap.add_argument("--model", type=str, required=True, help="Path to .joblib")
    ap.add_argument("--data", type=str, required=True, help="Path to CSV file")
    ap.add_argument("--target", type=str, default=None, help="Target column (defaults to last column)")
    return ap.parse_args()

def main():
    args = parse_args()
    model = joblib.load(args.model)

    df = pd.read_csv(args.data)
    target = args.target or df.columns[-1]
    y = df[target]
    X = df.drop(columns=[target])

    y_pred = model.predict(X)

    print("Accuracy:", accuracy_score(y, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    print("Precision (macro):", p, "Recall (macro):", r, "F1 (macro):", f1)
    print("\nClassification report:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    main()
