import os
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


DATASET = 'panda_school'
BATCH_SIZE = 32
IMAGE_SHAPE = (224, 224)

# Load the dataset
dataset_dir = pathlib.Path("./" + DATASET)
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(480, 640),  # Match your capture size
    batch_size=BATCH_SIZE,  # Adjust batch size as needed
    shuffle=True,  # Shuffle to ensure good train/valid split
    seed=123  # Ensures reproducibility
)

class_names = dataset.class_names
num_classes = len(dataset.class_names)
print("class_names (", num_classes, "):\n", class_names)

# Get dataset size
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)  # 70% for training
valid_size = int(0.15 * dataset_size)    # 15% Validation
test_size = dataset_size - train_size - valid_size  # Remaining 15% Test

# Split dataset manually
train_ds = dataset.take(train_size)
valid_ds = dataset.skip(train_size).take(valid_size)
test_ds = dataset.skip(train_size + valid_size)

## Pre-processing
def preprocess_data(image, label):
    image = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(image, tf.float32))
    image = tf.image.resize(image, IMAGE_SHAPE)
    return image, label

test_ds = test_ds.map(preprocess_data)

MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL = MODELS_DIR + 'mobilenet-' + DATASET
MODEL_KERAS = MODELS_DIR + 'mobilenet-quant-' + DATASET + '.keras'
model = tf.keras.models.load_model(MODEL)

# Generate predictions from test dataset
y_true = []
y_pred = []

for images, labels in test_ds:  # Iterate through batches
    predictions = model.predict(images)  # Get model predictions
    predicted_classes = np.argmax(predictions, axis=1)  # Get highest confidence index

    y_true.extend(labels.numpy())  # Convert tensor to NumPy array
    y_pred.extend(predicted_classes)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(f"confusion_matrix_{DATASET}.png")
#plt.show()

# Print Classification Report (Precision, Recall, F1-score)
print(classification_report(y_true, y_pred, target_names=class_names))
