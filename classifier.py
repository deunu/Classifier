import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths
base_dir = r'C:\\Users\\Asus\\OneDrive\\Desktop\\Archmage\\LeopardClassify AI model'
saved_model_dir = os.path.join(base_dir, 'saved_model')
batch_size = 32
img_height = 180
img_width = 180

# Create the saved_model directory if it doesn't exist
os.makedirs(saved_model_dir, exist_ok=True)

# Create an ImageDataGenerator and load images from directories
datagen = ImageDataGenerator(validation_split=0.2)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation'
)

# Define a simple model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_gen.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save the model in the SavedModel format
model.save(os.path.join(saved_model_dir, 'model.keras'))


