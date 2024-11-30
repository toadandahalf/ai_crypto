import random

import keras
from preprocessing import get_preprocessing

model = keras.models.load_model('main_model.keras')


def get_prediction(data):
    """[[close_went_up, close_went_down]]"""
    data = get_preprocessing(data)
    answer = model.predict(data)[0]
    if answer[0] > answer[1]:
        print('up')
    else:
        print('down')


# print(get_prediction([[random.randint(100, 150) for i in range(3)] for ii in range(14)]))
