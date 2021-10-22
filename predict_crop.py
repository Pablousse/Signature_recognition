from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv


model = keras.models.load_model("assets/crop_model")

train_datagen = ImageDataGenerator(rescale=1. / 255)

generator = train_datagen.flow_from_directory(
        "assets/prediction_cropped",
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
    filename = prediction[1].replace("test/", "").split("_")[0]
    if len(to_csv_format) > 0:
        if to_csv_format[-1][0] == filename and prediction[0][0] == 1:
            to_csv_format[-1][1] = 1
        elif not to_csv_format[-1][0] == filename:
            to_csv_format.append([filename, prediction[0][0]])
    else:
        to_csv_format.append([filename, prediction[0][0]])

for line in to_csv_format:
    print(line)

header = ["id", "Expected"]
with open('assets/sample_submission.csv', 'w') as f:

    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(header)
    write.writerows(to_csv_format)
