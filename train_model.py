import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time


# train_dir = "assets/a4_test/train"
train_dir = "assets/Cropped_image/train"

# Training data folder
# train_dir = os.path.join(dataset_folder, 'train')
# Test data folder
validation_dir = "assets/a4_test/validation"

# Directory with our training cat pictures
# train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
# train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
# validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')


# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling model
model.compile(
    # Choose the loss function
    loss='binary_crossentropy',
    # Choose your optimizer
    optimizer=RMSprop(learning_rate=1e-4),
    # Choose the metric the model will use to evaluate his learning
    metrics=['accuracy']
)


# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    # This is the source directory for training images
    train_dir,
    # All images will be resized to 150x150
    target_size=(150, 150),
    # Define how big are gonna be your batch.
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary'
)


# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Check if GPU is detected by tensorflow
print("GPU AVAILABLE: ", tf.config.list_physical_devices('GPU'))

print("--------- TRAINING START ---------")

start_time = time.time()

history = model.fit(
    train_generator,
    # 2000 images = batch_size * steps
    steps_per_epoch=90,
    epochs=50,
    # validation_data=validation_generator,
    # 1000 images = batch_size * steps
    # validation_steps=10,
    verbose=1
)

print("training DONE.")
print(f"Took {(time.time() - start_time)}s ")

model.save("assets/a4_model")

# print(model.predict_classes("assets/train/0a2c344efb5dd5b88450eec236a2aa3b_1.tif"))
# print(model.predict_classes("assets/train/0a2c344efb5dd5b88450eec236a2aa3b_2.tif"))
# print(model.predict_classes("assets/train/0a9aa4f1b00fb99492a54a99e70384be.tif"))
# print(model.predict_classes("assets/train/0a948131fe85c38152c0b9b22f7c09fc_2.tif"))
# print(model.predict_classes("assets/train/0a948131fe85c38152c0b9b22f7c09fc_3.tif"))
