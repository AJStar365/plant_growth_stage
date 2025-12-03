import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report

# -------------------------
# Directories
# -------------------------
train_dir = "dataset_resized/train"
val_dir = "dataset_resized/validation"

# -------------------------
# Data Generators
# -------------------------
# Reduced augmentation as per v2 fixes
train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode="categorical", shuffle=False
)

num_classes = train_gen.num_classes

# -------------------------
# Build Model (Transfer Learning - InceptionV3)
# -------------------------
base = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=predictions)

# Freeze base layers first
for layer in base.layers:
    layer.trainable = False

# -------------------------
# Compile Model
# -------------------------
# Increased learning rate to 0.001 as per v2 fixes
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------
# Callbacks
# -------------------------
callbacks_initial_train = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_plant_growth_stage.h5", save_best_only=True)
]

# Separate callbacks for fine-tuning to avoid OSError
callbacks_finetune = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("best_plant_growth_stage_finetuned.h5", save_best_only=True)
]

# -------------------------
# Initial Training (frozen base)
# -------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks_initial_train
)

# -------------------------
# Fine-tune (unfreeze top 2 inception blocks)
# -------------------------
# Unfreezing from layer 249 as per v2 fixes
for layer in base.layers[249:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks_finetune
)

# -------------------------
# Save Final Model
# -------------------------
model.save("plant_growth_stage.h5")
print("✅ Final model saved as plant_growth_stage.h5")

# -------------------------
# Evaluation
# -------------------------
val_preds = model.predict(val_gen)
y_pred = np.argmax(val_preds, axis=1)

print("\nClassification Report:")
print(classification_report(val_gen.classes, y_pred, target_names=list(val_gen.class_indices.keys())))