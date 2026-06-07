# train_model.py
# Builds, trains, and saves the CNN emotion classifier

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPooling2D,
                                     Dropout, Flatten, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)
from tensorflow.keras.utils import to_categorical

# ── REPRODUCIBILITY ─────────────────────────────────────────
# Setting a seed means you get same results every run
np.random.seed(42)
tf.random.set_seed(42)

# ── LOAD PREPROCESSED DATA ──────────────────────────────────
print("Loading preprocessed data...")
X_train = np.load('model/X_train.npy')
X_test  = np.load('model/X_test.npy')
y_train = np.load('model/y_train.npy')
y_test  = np.load('model/y_test.npy')
classes = np.load('model/classes.npy', allow_pickle=True)

NUM_CLASSES = len(classes)  # 7
print(f"Classes: {list(classes)}")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ── ONE-HOT ENCODE LABELS ────────────────────────────────────
# Convert numbers to arrays
# Example: 3 (happy) → [0, 0, 0, 1, 0, 0, 0]
# This is what the CNN output layer expects
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

# ── HANDLE CLASS IMBALANCE ───────────────────────────────────
# disgust has very few images vs happy which has many
# class weights tell the model to pay MORE attention to rare classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights (higher = rarer class):")
for i, cls in enumerate(classes):
    print(f"  {cls}: {class_weight_dict[i]:.2f}")

# ── BUILD THE CNN ARCHITECTURE ───────────────────────────────
print("\nBuilding CNN model...")

model = Sequential([

    # ── BLOCK 1 ──────────────────────────────────────────
    # First conv layer: 64 filters, each 3x3 pixels
    # Learns basic features: edges, corners, lines
    # padding='same' keeps output same size as input
    # input_shape: (48, 48, 1) = height, width, channels(grayscale=1)
    Conv2D(64, (3,3), activation='relu', padding='same',
           input_shape=(48, 48, 1)),
    BatchNormalization(),   # stabilize training
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),     # shrink from 48x48 → 24x24
    Dropout(0.25),          # randomly drop 25% of neurons

    # ── BLOCK 2 ──────────────────────────────────────────
    # 128 filters now: learns more complex patterns
    # eyes, nose shape, mouth curves
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),     # shrink from 24x24 → 12x12
    Dropout(0.25),

    # ── BLOCK 3 ──────────────────────────────────────────
    # 256 filters: learns high-level facial expressions
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),     # shrink from 12x12 → 6x6
    Dropout(0.25),

    # ── FULLY CONNECTED LAYERS ────────────────────────────
    Flatten(),              # convert 6x6x256 grid → 9216 numbers

    Dense(256, activation='relu'),  # learn combinations of features
    BatchNormalization(),
    Dropout(0.5),           # 50% dropout before final layer

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),

    # ── OUTPUT LAYER ──────────────────────────────────────
    # 7 neurons = 7 emotions
    # softmax converts raw scores to probabilities (sum = 1.0)
    Dense(NUM_CLASSES, activation='softmax')
])

# ── COMPILE THE MODEL ────────────────────────────────────────
# Adam optimizer: adjusts learning rate automatically
# categorical_crossentropy: standard loss for multi-class problems
# accuracy: the metric we care about
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary to see all layers and parameter counts
model.summary()

# ── CALLBACKS ────────────────────────────────────────────────
# These run automatically during training

# 1. Save best model automatically
checkpoint = ModelCheckpoint(
    'model/emotion_model.h5',
    monitor='val_accuracy',    # watch validation accuracy
    save_best_only=True,       # only save when it improves
    verbose=1
)

# 2. Stop training if no improvement for 15 epochs
# Prevents wasting time if model has plateaued
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# 3. Reduce learning rate if stuck
# If no improvement for 7 epochs, cut learning rate in half
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=0.00001,
    verbose=1
)

# ── TRAIN THE MODEL ──────────────────────────────────────────
print("\n" + "="*50)
print("Starting training...")
print("="*50)

history = model.fit(
    X_train, y_train_cat,
    batch_size=64,              # process 64 images at a time
    epochs=50,                  # maximum 50 passes over all data
    validation_data=(X_test, y_test_cat),
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ── EVALUATE FINAL MODEL ─────────────────────────────────────
print("\n" + "="*50)
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Final Test Loss:     {test_loss:.4f}")

# ── PLOT TRAINING HISTORY ────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'],
         label='Train Accuracy', color='blue')
ax1.plot(history.history['val_accuracy'],
         label='Val Accuracy', color='orange')
ax1.set_title('Model Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history['loss'],
         label='Train Loss', color='blue')
ax2.plot(history.history['val_loss'],
         label='Val Loss', color='orange')
ax2.set_title('Model Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model/training_history.png')
print("\nTraining history plot saved to model/training_history.png")

print("\n" + "="*50)
print("TRAINING COMPLETE")
print(f"Best model saved to: model/emotion_model.h5")
print("="*50)