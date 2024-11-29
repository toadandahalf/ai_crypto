import random

import keras
from preprocessing import get_preprocessing

model = keras.models.load_model('main_model.keras')


def get_prediction(data):
    data = get_preprocessing(data)
    return model.predict(data)


get_prediction([[random.randint(100, 150) for i in range(3)] for ii in range(14)])
