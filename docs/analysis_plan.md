# Plan: Analyze Training Parameters for Voxel51/DUTS

## Objective
Provide a highly specific, actionable training configuration for the Voxel51/DUTS dataset based on the user's provided metadata.

## Metadata Analysis
- **Dataset:** Voxel51/DUTS
- **Sample Count:** 500 images + 500 alphas (Exactly pairs) -> **Supervised Learning** is mandatory.
- **Average Size:** 379x327 pixels -> This is relatively low resolution. The `img_size` parameter in the model should not be set too high (e.g., 1024) to avoid upscaling artifacts during training. 384 or 512 is optimal.
- **Dataset Size:** 500 is a *small* dataset for deep learning. We need aggressive augmentation (already in `dataset.py`), lower learning rates, and more epochs to prevent overfitting.

## Task
Formulate a clear, direct response recommending the exact values the user should input into their "Training" UI in the Next.js frontend, explaining the *why* behind each choice based on the metadata.