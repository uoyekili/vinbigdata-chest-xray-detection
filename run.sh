#!/bin/bash
set -e

echo PYTHONPATH=.
python src/preprocess.py

echo "Training all models on Fold 0..."
python src/train.py --fold 0

echo ""
echo "Evaluating on local test set..."
python src/evaluate.py --fold 0

echo ""
echo "Generating submission..."
python src/inference.py

echo ""
echo "Complete!"
echo "Results: output/analysis/"
echo "Submission: output/submission.csv"
