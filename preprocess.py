# preprocess.py
# This script loads all FER-2013 images, converts them to
# numpy arrays, normalizes them, and saves them ready for training

import numpy as np
import os
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── CONFIGURATION ──────────────────────────────────────────
TRAIN_DIR = 'dataset/train'   # folder with 7 emotion subfolders
TEST_DIR  = 'dataset/test'    # folder with 7 emotion subfolders
IMG_SIZE  = 48                # each image is 48x48 pixels

# These are the exact folder names inside dataset/train
EMOTIONS = ['angry', 'disgust', 'fear', 'happy',
            'neutral', 'sad', 'surprise']

# ── FUNCTION TO LOAD IMAGES ─────────────────────────────────
def load_images(folder_path):
    """
    Goes through each emotion folder,
    loads every image, converts to grayscale array,
    and records its emotion label
    """
    images = []   # will hold pixel arrays
    labels = []   # will hold emotion names

    for emotion in EMOTIONS:
        emotion_path = os.path.join(folder_path, emotion)

        # Skip if folder doesn't exist
        if not os.path.exists(emotion_path):
            print(f"Warning: {emotion_path} not found, skipping")
            continue

        print(f"Loading {emotion}...")

        for img_file in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_file)

            try:
                # Open image and convert to grayscale
                # 'L' mode = grayscale (1 channel, 0-255 values)
                img = Image.open(img_path).convert('L')

                # Resize to exactly 48x48 in case any are different
                img = img.resize((IMG_SIZE, IMG_SIZE))

                # Convert image to numpy array (48x48 grid of numbers)
                img_array = np.array(img)

                images.append(img_array)
                labels.append(emotion)

            except Exception as e:
                # Skip corrupted images
                print(f"Skipping {img_path}: {e}")
                continue

    return np.array(images), np.array(labels)


# ── LOAD TRAINING AND TEST DATA ─────────────────────────────
print("=" * 50)
print("Loading training images...")
X_train, y_train = load_images(TRAIN_DIR)

print("\nLoading test images...")
X_test, y_test = load_images(TEST_DIR)


# ── NORMALIZE PIXEL VALUES ──────────────────────────────────
# Currently pixels are 0-255
# Divide by 255 to make them 0.0-1.0
# Neural networks train much faster on small numbers
X_train = X_train / 255.0
X_test  = X_test  / 255.0


# ── RESHAPE FOR CNN ─────────────────────────────────────────
# CNN expects shape: (samples, height, width, channels)
# channels = 1 because grayscale (color images would be 3)
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test  = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# ── ENCODE LABELS ───────────────────────────────────────────
# Convert emotion names to numbers
# angry=0, disgusted=1, fearful=2, happy=3, neutral=4, sad=5, surprised=6
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded  = le.transform(y_test)

# Save the label encoder classes so we can use them later
np.save('model/classes.npy', le.classes_)
print(f"\nEmotion classes saved: {le.classes_}")


# ── SAVE PROCESSED DATA ─────────────────────────────────────
# Save as .npy files so we don't have to reload images every time
print("\nSaving processed data...")
np.save('model/X_train.npy', X_train)
np.save('model/X_test.npy',  X_test)
np.save('model/y_train.npy', y_train_encoded)
np.save('model/y_test.npy',  y_test_encoded)


# ── PRINT SUMMARY ───────────────────────────────────────────
print("\n" + "=" * 50)
print("PREPROCESSING COMPLETE")
print("=" * 50)
print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Image shape      : {X_train.shape[1:]}")
print(f"Emotions         : {list(le.classes_)}")


# ── VISUALIZE SAMPLE IMAGES ─────────────────────────────────
print("\nGenerating sample visualization...")

fig, axes = plt.subplots(2, 7, figsize=(14, 4))
fig.suptitle('Sample images from each emotion class', fontsize=14)

for i, emotion in enumerate(le.classes_):
    # Find first image of this emotion in training set
    idx = np.where(y_train_encoded == i)[0][0]

    # Top row: training sample
    axes[0, i].imshow(X_train[idx].reshape(48, 48),
                      cmap='gray')
    axes[0, i].set_title(emotion, fontsize=9)
    axes[0, i].axis('off')

    # Bottom row: another sample of same emotion
    idx2 = np.where(y_train_encoded == i)[0][1]
    axes[1, i].imshow(X_train[idx2].reshape(48, 48),
                      cmap='gray')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('model/sample_images.png')
print("Sample visualization saved to model/sample_images.png")
