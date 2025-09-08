# src/cosmic_classifier/models.py
from __future__ import annotations

from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def build_model_candidates(random_state: int = 42) -> Dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=2000, n_jobs=None, random_state=random_state),
        "linear_svc": LinearSVC(random_state=random_state),
        "rf": RandomForestClassifier(n_estimators=300, random_state=random_state)
    }
