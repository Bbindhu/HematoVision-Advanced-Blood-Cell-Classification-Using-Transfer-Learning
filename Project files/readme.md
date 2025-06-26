project Executable files
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths and parameters
train_dir = "/path/to/train"
val_dir = "/path/to/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4  # or 8, depending on dataset

# Data generators with augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
).flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Base model
base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False

# Adding classification head
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)

# Optional fine-tuning
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_gen, epochs=5, validation_data=val_gen)
