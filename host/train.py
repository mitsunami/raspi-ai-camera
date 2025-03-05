import os
import tensorflow as tf
import pathlib
import numpy as np
import cv2

MODELS_DIR = 'models/'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

DATASET = 'panda_school'
MODEL = MODELS_DIR + 'mobilenet-' + DATASET
MODEL_KERAS = MODELS_DIR + 'mobilenet-quant-' + DATASET + '.keras'

BATCH_SIZE = 128
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

# Print dataset information
print(f"Total samples: {dataset_size}")
print(f"Train samples: {train_size}, Validation samples: {valid_size}, Test samples: {test_size}")


## Pre-processing and augmentation
def preprocess_data(image, label):
    # Set range
    image = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(image, tf.float32))
    # Resize image
    image = tf.image.resize(image, IMAGE_SHAPE)
    return image, label

train_ds = train_ds.map(preprocess_data)
valid_ds = valid_ds.map(preprocess_data)

# Add augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Randomly flip images
    tf.keras.layers.RandomRotation(0.1),  # Rotate images by 10% max
    tf.keras.layers.RandomZoom(0.05),  # Randomly zoom in by 5%
    tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),  # Move image
    #tf.keras.layers.RandomBrightness(0.1),  # Adjust brightness
    #tf.keras.layers.RandomContrast(0.1),  # Adjust contrast
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
valid_ds = valid_ds.prefetch(tf.data.experimental.AUTOTUNE)


## Keras Mobilenetv2 model for transfer learning
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

base_model =  MobileNetV2(weights='imagenet', include_top=False, input_shape=IMAGE_SHAPE+(3,))

# Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation=tf.nn.softmax)(x)

# Create the full model
float_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

float_model.summary()


## Train
EPOCHS = 50

float_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    baseline=0.7,
    min_delta=0.01,
    mode='max',
    patience=3,
    verbose=1,
    restore_best_weights=True,
    start_from_epoch=5,
)

history = float_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[callback]
)

float_model.save(MODEL)


## Visualize detections

import matplotlib.pyplot as plt

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
    cv2.imwrite(f'float_detections_{cls}_{score}.png', image_np)


# Visualize detection results for some images
for image, label in test_ds.unbatch().take(4):
    preprocessed = preprocess_image_visualization(image)
    predictions = detect_objects(float_model, preprocessed)
    cls, score = get_top_prediction(predictions)
    print(f'cls: {cls}, score: {score}')
    #assert score > 0.55
    visualize_detection(image, cls, score)
