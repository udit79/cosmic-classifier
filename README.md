# Cosmic Classifier

Train multiple classifiers on cosmicclassifierTraining.csv with an 80/20 split, TF-IDF text features, and full metrics.

## Run
# Install
pip install -r requirements.txt

# Train (auto-detects column types; target defaults to the last column)
python -m cosmic_classifier.train --data data/cosmicclassifierTraining.csv --target <TARGET_IF_KNOWN> --outdir .

# Evaluate a saved model
python -m cosmic_classifier.evaluate --model models/best_model.joblib --data data/cosmicclassifierTraining.csv --target <TARGET_IF_KNOWN>

Artifacts are written to ./models and ./metrics.
