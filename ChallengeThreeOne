# 1. Install/import needed libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

from sklearn.metrics import classification_report, confusion_matrix
# 2. Mount Google Drive if your dataset is stored there

from google.colab import drive
drive.mount('/content/drive')
# 3. Set your dataset path
# Change this path to wherever your Kaggle dataset folder is located.

DATASET_PATH = "/content/drive/MyDrive/chest_ct_scan_images"

# Example expected structure:
# chest_ct_scan_images/
#   train/
#     normal/
#     covid/
#     pneumonia/
#   test/
#     normal/
#     covid/
#     pneumonia/
# 4. Image settings

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10
# 5. Create image generators with preprocessing and augmentation

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
# 6. Load training and validation data

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)
# 7. Load test data

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)
# 8. Check class labels

print(train_generator.class_indices)

NUM_CLASSES = len(train_generator.class_indices)
# 9. Build CNN model

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.summary()
# 10. Compile model

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# 11. Train model

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)
# 12. Plot accuracy

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
# 13. Plot loss

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
# 14. Evaluate on test data

test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
# 15. Predictions

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
# 16. Classification report

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))
# 17. Confusion matrix

cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels,
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# 18. Save the model

model.save("medical_image_cnn_model.h5")
