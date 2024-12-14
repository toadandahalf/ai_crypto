import pandas as pd
import numpy as np
import keras
from constants_here import WINDOW_HEIGHT, PERIODS_RSI, PERIOD_SIGNAL_MACD, PERIOD_SHORT_MACD,\
    PERIOD_LONG_MACD, EPOCHS, WAY
from main_data_converter import main_data_converter

"""Источник данных - https://public.bybit.com"""

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)


x, y = main_data_converter(WAY, WINDOW_HEIGHT, PERIOD_LONG_MACD, PERIOD_SHORT_MACD, PERIOD_SIGNAL_MACD, PERIODS_RSI)
x_test, y_test, x_train, y_train = x[190000:], y[190000:], x[:190000], y[:190000]
print(x[:3], y[:3], sep='\n')

'''Модель тут'''
model_2 = keras.models.Sequential([
    keras.layers.Dense(12, activation='sigmoid'),
    keras.layers.Dense(20, activation='sigmoid'),
    keras.layers.Dense(20, activation='sigmoid'),
    keras.layers.Dense(2, activation='sigmoid')
])

chosen_model = model_2

chosen_model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

chosen_model.fit(x_train, y_train, epochs=EPOCHS)

chosen_model.evaluate(x_test, y_test)

chosen_model.save('main_model.keras')

prediction = chosen_model.predict(x)

print(prediction[0], '- пример предсказанных данных')

minus, plus = 0, 0
for i in prediction:
    if i[0][0] > 0.5:
        plus += 1
        print(1, end='')
    else:
        minus += 1
        print(0, end='')
print()
print(plus, minus, 'plus, minus')
