import cv2
import os, logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import boto3
import matplotlib.pyplot as plt
#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
#from keras.utils import to_categorical
#from keras.callbacks import EarlyStopping
#from keras.models import Model

IMG_SIZE = 96
# IMG_SIZE = 224
NUM_ETHNICITIES = 5
NUM_GENDERS = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "UTKFace")
FILE_DIR = os.path.join(BASE_DIR, "data", "files")
MODEL_DIR = os.path.join(BASE_DIR, "data", "model")
S3_BUCKET_NAME = "krkprojects"
S3_MODEL_DIR = "model"
PROJECT_NAME = "CNN_Project_UTKFace"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.FileHandler('CNN_Project/project.log'), logging.StreamHandler()])

def load_data(data):
    images = []
    ages = []
    genders = []
    ethnicities = []    

    logging.info("Loading data")

    if not os.path.exists(data):
        logging.error("Data not found")
        raise Exception("Data not found")
    
    count = 0    
    for file in os.listdir(data):
        if len(file.split("_")) < 3:
            continue
        if not file.endswith(".jpg"):
            continue
        count += 1
        if count < 23706:
            age, gender, ethnicitie = map(int, file.split("_")[:3])

            img = cv2.imread(os.path.join(data, file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            images.append(img)
            ages.append(age)
            genders.append(gender)
            ethnicities.append(ethnicitie)

    return images, ages, genders, ethnicities

# def cnn_model():
#     inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

#     x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)

#     x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)

#     x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)

#     x = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.MaxPooling2D(2, 2)(x)

#     x = tf.keras.layers.Flatten()(x)

#     age_branch = tf.keras.layers.Dense(128, activation="relu")(x)
#     age_branch = tf.keras.layers.Dropout(0.5)(age_branch)
#     age_output = tf.keras.layers.Dense(1, activation="linear", name="age_output")(age_branch)

#     gender_branch = tf.keras.layers.Dense(128, activation="relu")(x)
#     gender_branch = tf.keras.layers.Dropout(0.5)(gender_branch)
#     gender_output = tf.keras.layers.Dense(2, activation="softmax", name="gender_output")(gender_branch)

#     ethnicity_branch = tf.keras.layers.Dense(128, activation="relu")(x)
#     ethnicity_branch = tf.keras.layers.Dropout(0.5)(ethnicity_branch)
#     ethnicity_output = tf.keras.layers.Dense(5, activation="softmax", name="ethnicity_output")(ethnicity_branch)

#     model = tf.keras.models.Model(inputs=inputs, outputs=[age_output, gender_output, ethnicity_output])

 
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
#         loss={
#             "age_output": tf.keras.losses.MeanSquaredError(),
#             "gender_output": tf.keras.losses.CategoricalCrossentropy(),
#             "ethnicity_output": tf.keras.losses.CategoricalCrossentropy()
#         },
#         loss_weights={ 
#             "age_output": 0.1,  
#             "gender_output": 1.0,
#             "ethnicity_output": 1.0
#         },
#         metrics={
#             "age_output": tf.keras.metrics.MeanSquaredError(),
#             "gender_output": tf.keras.metrics.CategoricalAccuracy(),
#             "ethnicity_output": tf.keras.metrics.CategoricalAccuracy()
#         }
#     )
    
#     return model

def mobilenetv3_model(trainable_base=False):
    base_model = tf.keras.applications.MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = trainable_base 

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)  
    x = base_model(x, training=False)  
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  
    x = tf.keras.layers.BatchNormalization()(x)

    age_branch = tf.keras.layers.Dense(128, activation="relu")(x)
    age_branch = tf.keras.layers.Dropout(0.3)(age_branch)  
    age_output = tf.keras.layers.Dense(1, activation="linear", name="age_output")(age_branch)

    gender_branch = tf.keras.layers.Dense(128, activation="relu")(x)
    gender_output = tf.keras.layers.Dense(NUM_GENDERS, activation="softmax", name="gender_output")(gender_branch)

    ethnicity_branch = tf.keras.layers.Dense(128, activation="relu")(x)
    ethnicity_output = tf.keras.layers.Dense(NUM_ETHNICITIES, activation="softmax", name="ethnicity_output")(ethnicity_branch)

    model = tf.keras.models.Model(inputs=inputs, outputs=[age_output, gender_output, ethnicity_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss={
            "age_output": tf.keras.losses.Huber(delta=10),
            "gender_output": tf.keras.losses.CategoricalCrossentropy(),
            "ethnicity_output": tf.keras.losses.CategoricalCrossentropy()
        },
        metrics={
            "age_output": tf.keras.metrics.MeanSquaredError(),
            "gender_output": tf.keras.metrics.CategoricalAccuracy(),
            "ethnicity_output": tf.keras.metrics.CategoricalAccuracy()
        }
    )

    return model


if __name__ == "__main__":

    logging.info("Training model")
    # Check if model already exists
    if not os.path.exists(os.path.join(MODEL_DIR, "cnn_model.keras")):
        
        if len(os.listdir(FILE_DIR)) == 4:
            logging.info("Data already exists")
            logging.info("Loading data")
            X = np.load(os.path.join(FILE_DIR, "X.npy"))
            ages = np.load(os.path.join(FILE_DIR, "ages.npy"))
            genders = np.load(os.path.join(FILE_DIR, "genders.npy"))
            ethnicities = np.load(os.path.join(FILE_DIR, "ethnicities.npy"))
        else:
            logging.info("Getting data")
            X, ages, genders, ethnicities = load_data(DATA_DIR)

            logging.info("Converting data to numpy array")
            X = np.array(X)
            ages = np.array(ages)
            genders = np.array(genders)
            ethnicities = np.array(ethnicities)

            logging.info("Saving data")
            np.save(os.path.join(FILE_DIR, "X.npy"), X)
            np.save(os.path.join(FILE_DIR, "ages.npy"), ages)
            np.save(os.path.join(FILE_DIR, "genders.npy"), genders)
            np.save(os.path.join(FILE_DIR, "ethnicities.npy"), ethnicities)

        logging.info("Splitting data")
        X_train, X_test, y_train_age, y_test_age, y_train_gender, y_test_gender, y_train_ethnicity, y_test_ethnicity = train_test_split(X, ages, genders, ethnicities, test_size=0.2, random_state=42)

        logging.info("Converting data to categorical")
        y_train_gender = tf.keras.utils.to_categorical(y_train_gender, 2)
        y_test_gender = tf.keras.utils.to_categorical(y_test_gender, 2)

        y_train_ethnicity = tf.keras.utils.to_categorical(y_train_ethnicity, 5)
        y_test_ethnicity = tf.keras.utils.to_categorical(y_test_ethnicity, 5)

        logging.info("Setting up model")
        # model = cnn_model()

        model = mobilenetv3_model(trainable_base=False)

        print(model.summary())
        logging.info("Setting up callbacks")
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
        logging.info("Fitting/Training model")
        history = model.fit(
            X_train, {"age_output": y_train_age, "gender_output": y_train_gender, "ethnicity_output": y_train_ethnicity}, 
            validation_data=(X_test, {"age_output": y_test_age, "gender_output": y_test_gender, "ethnicity_output": y_test_ethnicity}), 
            epochs=20, 
            batch_size=32, 
            verbose=1,
            callbacks=[early_stop, learning_rate_reduction]
        )
        model = mobilenetv3_model(trainable_base=True)
        model.fit(
            X_train, {"age_output": y_train_age, "gender_output": y_train_gender, "ethnicity_output": y_train_ethnicity}, 
            validation_data=(X_test, {"age_output": y_test_age, "gender_output": y_test_gender, "ethnicity_output": y_test_ethnicity}), 
            epochs=20, 
            batch_size=32, 
            verbose=1,
            callbacks=[early_stop, learning_rate_reduction]
        )

        logging.info("Plotting results for overall training and validation loss")
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        logging.info("Plotting results for gender and ethnicity training and validation accuracy")
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['ethnicity_output_categorical_accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_ethnicity_output_categorical_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Ethnicity Classification Accuracy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['gender_output_categorical_accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_gender_output_categorical_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Gender Classification Accuracy')
        plt.legend()
        plt.grid(True)

        plt.show()

        logging.info("Plotting results for age training and validation mean squared error")
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['age_output_mean_squared_error'], label='Training MSE')
        plt.plot(history.history['val_age_output_mean_squared_error'], label='Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Age Prediction Error Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        logging.info("Plotting results for learning rate")
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['learning_rate'], label='Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        logging.info("Saving model")
        tf.keras.models.save_model(model, os.path.join(MODEL_DIR, "cnn_model.keras"))

        logging.info("AWS Upload Started")
        s3_client = boto3.client('s3')
        s3_client.upload_file(os.path.join(MODEL_DIR, "cnn_model.keras"), S3_BUCKET_NAME, os.path.join(PROJECT_NAME,S3_MODEL_DIR, "cnn_model.keras"))
        logging.info("AWS Upload Completed")

        logging.info("AWS Download Started")
        s3_client.download_file(S3_BUCKET_NAME, os.path.join(PROJECT_NAME,S3_MODEL_DIR, "cnn_model.keras"), os.path.join(BASE_DIR, "Web", "backend", "app", "data", "cnn_model.keras"))
        logging.info("AWS Download Completed")
        
    else:
        def detect_age_gender_ethnicity():
            model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))
            labels_gender = ["Male", "Female"]
            labels_ethnicity = ["White", "Black", "Asian", "Indian", "Others"]
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    face = np.expand_dims(face / 255.0, axis=0)
                    age, gender, ethnicity = model.predict(face)
                    age = int(age[0][0])
                    gender_label = labels_gender[np.argmax(gender)]
                    ethnicity_label = labels_ethnicity[np.argmax(ethnicity)]
                    text = f"{gender_label}, {age} yrs, {ethnicity_label}"
                    color = (0, 255, 0) if gender_label == "Male" else (255, 0, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("Age, Gender & Ethnicity Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()

        detect_age_gender_ethnicity()

