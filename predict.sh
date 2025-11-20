#!/bin/bash

# AeroEyes Prediction Pipeline - Zalo AI Challenge 2025
# This script runs the streaming predictor and saves results to /result/submission.json

set -e

echo "🚀 Starting AeroEyes Prediction Pipeline..."

# Create results directory if not exists
mkdir -p /result

# Run prediction with Python3
python3 /code/predict.py

# Verify output exists
if [ -f "/result/submission.json" ]; then
    echo "✅ Success! Output saved to /result/submission.json"
    echo "📋 Output preview:"
    head -c 500 /result/submission.json
    echo -e "\n..."
else
    echo "❌ Error: Output file not found at /result/submission.json"
    exit 1
fi
