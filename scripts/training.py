import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

base_dir = Path('processed_datasets')

train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize the images
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

image_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    base_dir / 'training_data',
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    base_dir / 'validation_data',
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    base_dir / 'testing_data',
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Fourth convolutional block (added)
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten the output of the convolutions
    layers.Flatten(),

    # First fully connected block
    layers.Dense(256, activation='relu'),  # Increased number of neurons
    layers.Dropout(0.5),

    # Second fully connected block
    layers.Dense(128, activation='relu'),  # Increased number of neurons
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(train_generator.num_classes, activation='softmax')
])

optimizer = optimizers.Adam(learning_rate=0.00005)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

model.save('my_model.h5')

print("Model Training Completed!")
