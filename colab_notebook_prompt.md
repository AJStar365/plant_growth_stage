# Plant Growth Stage Classifier: Assignment Notebook

Hello! This notebook is structured to help you complete your academic assignment on plant growth stage classification. It incorporates a machine learning model (transfer learning with InceptionV3) and includes sections for data visualization using Matplotlib and model evaluation with Scikit-learn.

---

## 1. Notebook Setup and Dependencies

This section covers the initial setup for your Colab environment, including necessary library imports and mounting Google Drive to access your dataset.

```python
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Mount Google Drive to access the dataset
from google.colab import drive
drive.mount('/content/drive')
```

---

## 2. Data Preparation

Here, you will define the paths to your dataset stored on Google Drive and set up common image processing parameters like target size and batch size.

Your dataset is expected to be located on your Drive at `My Drive/dataset_resized`, with `train` and `validation` subdirectories. Each of these subdirectories should contain 6 classes of plant images.

```python
# Define paths to your dataset on Google Drive
# IMPORTANT: Adjust this path if your dataset is in a different location
base_data_dir = '/content/drive/My Drive/dataset_resized'
train_dir = os.path.join(base_data_dir, 'train')
val_dir = os.path.join(base_data_dir, 'validation')

# Set image dimensions and batch size
target_size = (224, 224) # InceptionV3 input size
batch_size = 32
```

---

## 3. Data Loading with `ImageDataGenerator`

This section sets up `ImageDataGenerator` instances for both training and validation datasets. The generators will handle loading images from disk, resizing them to the `target_size`, and scaling pixel values. For initial diagnosis, we will only apply pixel rescaling and no heavy augmentation.

```python
# Create ImageDataGenerator for the training set (only rescaling for now)
train_datagen = ImageDataGenerator(rescale=1./255)

# Create ImageDataGenerator for the validation set (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators using flow_from_directory
print("Preparing training data generator...")
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    # shuffle=True for training data
)

print("\nPreparing validation data generator...")
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False # Keep validation data in order for evaluation
)

# Determine the number of classes from the training generator
num_classes = train_gen.num_classes
print(f"\nDetected {num_classes} classes.")
```

---

## 4. Model Architecture: InceptionV3 Transfer Learning

Here, we will build our classification model using transfer learning with InceptionV3. The pre-trained InceptionV3 model will serve as a powerful feature extractor, and we will add a custom classification head tailored for our specific plant growth stages.

```python
# Load the InceptionV3 model pre-trained on ImageNet
# include_top=False: don't include the ImageNet classification layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))

# Freeze the layers of the base model
# This prevents their weights from being updated during the initial training phase
for layer in base_model.layers:
    layer.trainable = False

# Add a new classification head on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x) # Reduces spatial dimensions
x = Dropout(0.3)(x)             # Regularization to prevent overfitting
predictions = Dense(num_classes, activation='softmax')(x) # Output layer for 6 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()
```

---

## 5. Initial Model Training (Frozen Base)

In this phase, only the newly added classification head will be trained. The weights of the InceptionV3 base model remain frozen. This allows the head to learn how to classify our specific plant stages using the powerful features extracted by the pre-trained base.

```python
# Compile the model for the initial training phase
# Using Adam optimizer with a slightly higher learning rate for the head
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks for initial training
# EarlyStopping: stops training if validation accuracy doesn't improve
# ModelCheckpoint: saves the best model weights based on validation accuracy
callbacks_initial_train = [
    EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True),
    ModelCheckpoint('best_plant_growth_stage_initial.h5', save_best_only=True, monitor='val_accuracy', mode='max')
]

# Train the model
print("\n--- Starting Initial Training (Frozen Base) ---")
history_initial = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20, # Number of epochs for initial training
    callbacks=callbacks_initial_train
)
```

---

## 6. Fine-Tuning Phase (Unfrozen Top Layers)

After the initial training, we will unfreeze a portion of the base model's top layers and continue training with a very low learning rate. This allows the model to fine-tune its high-level feature extraction specifically for our dataset, potentially leading to higher accuracy.

```python
# Unfreeze the top layers of the base model (e.g., from layer 249 onwards for InceptionV3)
# This makes them trainable
for layer in base_model.layers[249:]:
    layer.trainable = True

# Re-compile the model with a much lower learning rate for fine-tuning
# This prevents destroying the carefully learned pre-trained weights
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks for fine-tuning
callbacks_finetune = [
    EarlyStopping(patience=10, monitor='val_accuracy', restore_best_weights=True),
    ModelCheckpoint('best_plant_growth_stage_finetuned.h5', save_best_only=True, monitor='val_accuracy', mode='max')
]

# Continue training the model (fine-tuning)
print("\n--- Starting Fine-Tuning (Unfrozen Top Layers) ---")
history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15, # Number of additional epochs for fine-tuning
    callbacks=callbacks_finetune
)
```

---

## 7. Model Evaluation and Visualization

This final section covers evaluating the performance of your trained model using a classification report and visualizing the training progress with Matplotlib plots.

```python
# Save the final trained model
model.save("plant_growth_stage_final.h5")
print("\n✅ Final model saved as plant_growth_stage_final.h5")

# --- Scikit-learn Classification Report ---
print("\n--- Model Evaluation (Classification Report) ---")
# Get predictions on the validation set
val_gen.reset() # Important to reset generator before prediction
val_preds = model.predict(val_gen)
y_pred = np.argmax(val_preds, axis=1) # Convert probabilities to class labels

# Get true labels
y_true = val_gen.classes

# Get class names
class_names = list(val_gen.class_indices.keys())

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# --- Matplotlib Training History Visualization ---
print("\n--- Training History Visualization ---")

def plot_training_history(history_initial, history_finetune):
    # Combine history objects for plotting
    acc = history_initial.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history_initial.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history_initial.history['loss'] + history_finetune.history['loss']
    val_loss = history_initial.history['val_loss'] + history_finetune.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history_initial.history['accuracy']) - 1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)


    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history_initial.history['accuracy']) - 1, color='r', linestyle='--', label='Fine-tuning Start')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_training_history(history_initial, history_finetune)
```