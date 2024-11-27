import keras
from preprocessing import get_preprocessing

model = keras.models.load_model('main_model.keras')


def get_prediction(fresh_list, previous_list):

    return model.predict()
