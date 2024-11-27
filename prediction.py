import keras

model = keras.models.load_model('main_model.keras')


def get_prediction(input_list):
    return model.predict()