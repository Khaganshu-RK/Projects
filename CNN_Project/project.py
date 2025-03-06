import cv2
import os, logging
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Model

IMG_SIZE = 224

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "UTKFace")
FILE_DIR = os.path.join(BASE_DIR, "data", "files")
MODEL_DIR = os.path.join(BASE_DIR, "data", "model")

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
        if count < 20000:
            age, gender, ethnicitie = map(int, file.split("_")[:3])

            img = cv2.imread(os.path.join(data, file))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            images.append(img)
            ages.append(age)
            genders.append(gender)
            ethnicities.append(ethnicitie)

    return images, ages, genders, ethnicities

def cnn_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = MaxPooling2D(2, 2)(x)

    x = Flatten()(x)

    age_branch = Dense(64, activation="relu")(x)
    age_output = Dense(1, activation="linear", name="age_output")(age_branch)

    gender_branch = Dense(64, activation="relu")(x)
    gender_output = Dense(2, activation="softmax", name="gender_output")(gender_branch)

    ethnicity_branch = Dense(64, activation="relu")(x)
    ethnicity_output = Dense(5, activation="softmax", name="ethnicity_output")(ethnicity_branch)

    model = Model(inputs=inputs, outputs=[age_output, gender_output, ethnicity_output])

    model.compile(
        optimizer="adam",
        loss={
            "age_output": keras.losses.MeanSquaredError(),
            "gender_output": keras.losses.CategoricalCrossentropy(),
            "ethnicity_output": keras.losses.CategoricalCrossentropy()
        },
        metrics={
            "age_output": keras.metrics.MeanSquaredError(),
            "gender_output": keras.metrics.Accuracy(),
            "ethnicity_output": keras.metrics.Accuracy()
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
        y_train_gender = to_categorical(y_train_gender, 2)
        y_test_gender = to_categorical(y_test_gender, 2)

        y_train_ethnicity = to_categorical(y_train_ethnicity, 5)
        y_test_ethnicity = to_categorical(y_test_ethnicity, 5)

        logging.info("Setting up model")
        model = cnn_model()

        print(model.summary())
        logging.info("Setting up callbacks")
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        logging.info("Fitting/Training model")
        history = model.fit(
            X_train, {"age_output": y_train_age, "gender_output": y_train_gender, "ethnicity_output": y_train_ethnicity}, 
            validation_data=(X_test, {"age_output": y_test_age, "gender_output": y_test_gender, "ethnicity_output": y_test_ethnicity}), 
            epochs=10, 
            batch_size=32, 
            verbose=1,
            callbacks=[early_stop]
        )
        logging.info("Saving model")
        keras.models.save_model(model, os.path.join(MODEL_DIR, "cnn_model.keras"))

    else:
        def detect_age_gender_ethnicity():
            model = keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))
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

