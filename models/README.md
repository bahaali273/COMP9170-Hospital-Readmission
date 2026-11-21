
# Trained Models

This directory contains trained machine learning model checkpoints.

## GraphSAGE

- File: `graphsage_checkpoint.pt`
- Architecture: Two-layer GraphSAGE
- Input features: 133
- Hidden units: 64
- Training epochs: 20
- Validation-selected threshold: 0.53

The checkpoint contains trained model weights and configuration.

Reproducing predictions also requires the training-fitted
StandardScaler, the training reference features, and the
patient-similarity graph construction procedure.

This model is for research and educational use only.
