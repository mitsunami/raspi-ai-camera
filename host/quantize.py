# Observe that we are using training part of the dataset as representative dataset.
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Generator
import pathlib
import cv2

DATASET = 'panda_school'

BATCH_SIZE = 32
IMAGE_SHAPE = (224, 224)

# Load the RPS dataset
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

dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% for training
valid_size = dataset_size - train_size  # 20% for testing

train_ds = dataset.take(train_size)
valid_ds = dataset.skip(train_size)

def preprocess_data(image, label):
    # Set range
    image = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(image, tf.float32))
    # Resize image
    image = tf.image.resize(image, IMAGE_SHAPE)
    return image, label

train_ds = train_ds.map(preprocess_data)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # Randomly flip images
    tf.keras.layers.RandomRotation(0.2),  # Rotate images by 20% max
    #tf.keras.layers.RandomZoom(0.1),  # Randomly zoom in by 20%
    #tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),  # Move image
    #tf.keras.layers.RandomBrightness(0.1),  # Adjust brightness
    #tf.keras.layers.RandomContrast(0.1),  # Adjust contrast
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

n_iter=10

# Create representative dataset generator
def get_representative_dataset() -> Generator:
    """A function that loads the dataset and returns a representative dataset generator.

    Returns:
        Generator: A generator yielding batches of preprocessed images.
    """
    dataset = train_ds

    def representative_dataset() -> Generator:
        """A generator function that yields batch of preprocessed images.

        Yields:
            A batch of preprocessed images.
        """
        for _ in range(n_iter):
            yield dataset.take(1).get_single_element()[0].numpy()

    return representative_dataset

# Create a representative dataset generator
representative_dataset_gen = get_representative_dataset()


import model_compression_toolkit as mct
from model_compression_toolkit.core import QuantizationErrorMethod

# Specify the IMX500-v1 target platform capability (TPC)
tpc = mct.get_target_platform_capabilities("tensorflow", 'imx500', target_platform_version='v1')

# Set the following quantization configurations:
# Choose the desired QuantizationErrorMethod for the quantization parameters search.
# Enable weights bias correction induced by quantization.
# Enable shift negative corrections for improving 'signed' non-linear functions quantization (such as swish, prelu, etc.)
# Set the threshold to filter outliers with z_score of 16.
q_config = mct.core.QuantizationConfig(activation_error_method=QuantizationErrorMethod.MSE,
                                       weights_error_method=QuantizationErrorMethod.MSE,
                                       weights_bias_correction=True,
                                       shift_negative_activation_correction=True,
                                       z_threshold=16)

ptq_config = mct.core.CoreConfig(quantization_config=q_config)


MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL = MODELS_DIR + 'mobilenet-' + DATASET
MODEL_KERAS = MODELS_DIR + 'mobilenet-quant-' + DATASET + '.keras'
float_model = tf.keras.models.load_model(MODEL)


quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(
    in_model=float_model,
    representative_data_gen=representative_dataset_gen,
    core_config=ptq_config,
    target_platform_capabilities=tpc)


# Export the quantized model
mct.exporter.keras_export_model(model=quantized_model, save_model_path=MODEL_KERAS)


## Visualize detections

import matplotlib.pyplot as plt
# Load the test part of the dataset
test_ds = valid_ds

# Preprocess the input image for inference
def preprocess_image_visualization(image):
    resized = tf.image.resize(image, (224, 224))
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
    return preprocessed

# Perform detection on the input image
def detect_objects(model, tensor):
    tensor = np.expand_dims(tensor, axis=0)
    predictions = model.predict(tensor)
    return predictions

# Get the class label and confidence score of the detected objects
def get_top_prediction(predictions):
    top_idx = np.argsort(predictions)[0][-1]
    top_score = predictions[0][top_idx]
    top_class = class_names[top_idx]
    return top_class, top_score

# Visualize the detections
def visualize_detection(image, cls, score):
    image_np = image.numpy().astype(np.uint8)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.putText(image_np, f'{cls}: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imwrite(f'quant_detections_{cls}_{score}.png', image_np)


# Visualize detection results for some images
for image, label in test_ds.unbatch().take(4):
    preprocessed = preprocess_image_visualization(image)
    predictions = detect_objects(quantized_model, preprocessed)
    cls, score = get_top_prediction(predictions)
    print(f'cls: {cls}, score: {score}')
    #assert score > 0.55
    visualize_detection(image, cls, score)
