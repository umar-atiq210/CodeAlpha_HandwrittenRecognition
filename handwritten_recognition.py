import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF info logs

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

print("=" * 50)
print("  HANDWRITTEN CHARACTER RECOGNITION — TASK 3")
print("=" * 50)
print(f"\nTensorFlow version : {tf.__version__}")

# ─────────────────────────────────────────
#  Load MNIST Dataset (built-in, no download)
# ─────────────────────────────────────────
print("\nLoading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training samples  : {X_train.shape[0]}")
print(f"Testing  samples  : {X_test.shape[0]}")
print(f"Image shape       : {X_train.shape[1:]}  (28x28 pixels)")
print(f"Classes           : 0-9 digits")

# ─────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────
# Normalize pixel values 0-255 -> 0.0-1.0
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Reshape for CNN: (samples, height, width, channels)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1,  28, 28, 1)

# One-hot encode labels (e.g. 3 -> [0,0,0,1,0,0,0,0,0,0])
y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

# ─────────────────────────────────────────
#  Build CNN Model
# ─────────────────────────────────────────
print("\nBuilding CNN model...")

model = models.Sequential([
    # Block 1 - detect simple edges/patterns
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(28, 28, 1), padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2 - detect complex shapes
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Flatten + Fully Connected
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    # Output - 10 classes (digits 0-9)
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────
#  Train Model
# ─────────────────────────────────────────
print("\nTraining model (this takes 2-4 minutes)...")

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ─────────────────────────────────────────
#  Evaluate
# ─────────────────────────────────────────
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

print(f"\n{'─' * 50}")
print(f"Test Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")
print(f"\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=[str(i) for i in range(10)]
))

# ─────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Handwritten Digit Recognition — CNN Results',
             fontsize=14, fontweight='bold')

# 1. Training history — Accuracy
axes[0, 0].plot(history.history['accuracy'],
                label='Train Accuracy', color='#185FA5', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'],
                label='Val Accuracy',   color='#2ecc71', linewidth=2)
axes[0, 0].set_title('Training vs Validation Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Training history — Loss
axes[0, 1].plot(history.history['loss'],
                label='Train Loss', color='#e74c3c', linewidth=2)
axes[0, 1].plot(history.history['val_loss'],
                label='Val Loss',   color='#f39c12', linewidth=2)
axes[0, 1].set_title('Training vs Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=range(10), yticklabels=range(10))
axes[1, 0].set_title('Confusion Matrix (0-9 Digits)')
axes[1, 0].set_xlabel('Predicted Digit')
axes[1, 0].set_ylabel('Actual Digit')

# 4. Accuracy text panel
axes[1, 1].axis('off')
axes[1, 1].text(0.5, 0.5,
    f'Test Accuracy\n{test_acc * 100:.2f}%\n\nSee sample_predictions.png\nfor visual results',
    ha='center', va='center', fontsize=16, fontweight='bold',
    transform=axes[1, 1].transAxes,
    color='#185FA5')

plt.tight_layout()
plt.savefig('handwritten_recognition_results.png', dpi=150, bbox_inches='tight')
plt.show()

# 5. Sample predictions — separate figure
sample_fig = plt.figure(figsize=(8, 8))
sample_fig.suptitle('Sample Predictions (Green=Correct, Red=Wrong)',
                    fontsize=11)
for i in range(25):
    ax = sample_fig.add_subplot(5, 5, i + 1)
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    ax.set_title(f'P:{y_pred[i]} A:{y_test[i]}',
                 color=color, fontsize=8, fontweight='bold')
    ax.axis('off')
sample_fig.tight_layout()
sample_fig.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlots saved:")
print("  -> handwritten_recognition_results.png")
print("  -> sample_predictions.png")
print("\n" + "=" * 50)
print("         TASK 3 COMPLETE")
print("=" * 50)