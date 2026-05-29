# Handwritten Character Recognition — CodeAlpha ML Internship Task 3

Recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN)
trained on the MNIST dataset. Achieves ~99% test accuracy.

## Model Architecture
- Conv2D (32 filters) → BatchNorm → MaxPool → Dropout
- Conv2D (64 filters) → BatchNorm → MaxPool → Dropout
- Dense (128) → BatchNorm → Dropout
- Dense (10, softmax) — output layer

## Dataset
MNIST — 70,000 handwritten digit images (60k train / 10k test), 28×28 pixels

## How to Run

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
python handwritten_recognition.py
```

## Output
- Terminal: classification report per digit (0–9)
- `handwritten_recognition_results.png` — training curves + confusion matrix
- `sample_predictions.png` — 25 sample predictions (green=correct, red=wrong)
