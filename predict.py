from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import csv


print("------------------------TEST 1------------------------")

model = keras.models.load_model("assets/a4_model")

print("------------------------TEST 456------------------------")

train_datagen = ImageDataGenerator(rescale=1./255)

# # Flow training images in batches of 20 using train_datagen generator
# train_generator = train_datagen.flow_from_directory(
#     # This is the source directory for training images
#     "assets/test",
#     # All images will be resized to 150x150
#     target_size=(150, 150),
#     # Define how big are gonna be your batch.
#     batch_size=20,
#     # Since we use binary_crossentropy loss, we need binary labels
#     class_mode='binary'
# )
# print(model.predict(train_generator))

print("------------------------TEST 3------------------------")

generator = train_datagen.flow_from_directory(
        "assets/Prediction",
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary",  # this means our generator will only yield batches of data, no labels
        shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
# the predict_generator method returns the output of a model, given
# a generator that yields batches of numpy data
bottleneck_features_train = (model.predict(generator) > 0.5).astype(int)

predictions = list(zip(bottleneck_features_train, generator.filenames))

to_csv_format = []

for prediction in predictions:
    to_csv_format.append([os.path.splitext(prediction[1])[0].replace("test/",""), prediction[0][0]])

for line in to_csv_format:
    print(line)

header = ["id", "Expected"]
with open('assets/sample_submission.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(header)
    write.writerows(to_csv_format)