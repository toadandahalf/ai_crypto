import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
import os
import joblib
from indicators import macd, rsi, close_trend_heatmap, macd_to_zero_heatmap, macd_val_to_signal_heatmap
from constants_here import SYMBOL, WINDOW_HEIGHT, PERIODS_RSI, PERIOD_SIGNAL_MACD, PERIOD_SHORT_MACD, PERIOD_LONG_MACD, \
    EPOCHS, WAY, STEP_BACK
from deleter import deleter

file_names = os.listdir(WAY)

if file_names[0] == '.ipynb_checkpoints':
    file_names.pop(0)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)


# source = 'https://public.bybit.com'

def data_converter(csv_names, window_height, way):
    '''из первого датасета берет с конца данные размера окна'''

    raw_list = []
    features = []
    labels = []

    with open(f'{way}/{csv_names[0]}', 'r') as first_to_pop:
        raw_list.extend(
            list(map(lambda x: x.rstrip().split(','), first_to_pop.readlines()[- (STEP_BACK + 1):]))
        )
    csv_names.pop(0)

    for i in csv_names:
        with open(f'{way}/{i}', 'r') as csv_file:
            raw_list.extend(list(map(lambda x: x.rstrip().split(','), csv_file.readlines()[1:]))
                            )

    raw_list = pd.DataFrame(raw_list, columns=['timestamp', 'symbol', 'period', 'open', 'high', 'low', 'close'])
    del raw_list['timestamp']
    del raw_list['symbol']
    del raw_list['period']
    del raw_list['open']

    raw_list = macd(raw_list, period_long=PERIOD_LONG_MACD, period_short=PERIOD_SHORT_MACD,
                    period_signal=PERIOD_SIGNAL_MACD)
    raw_list = rsi(raw_list, periods=PERIODS_RSI)
    raw_list = macd_val_to_signal_heatmap(raw_list, 'macd_val', 'macd_signal_line')
    raw_list = macd_to_zero_heatmap(raw_list, 'macd_val')
    raw_list = close_trend_heatmap(raw_list)
    raw_list = raw_list.dropna()

    scaler = MinMaxScaler()
    scaler.fit(raw_list)
    raw_list = scaler.transform(raw_list)
    joblib.dump(scaler, 'minmax_scaler.pkl')

    raw_list = pd.DataFrame(raw_list,
                            columns=['high', 'low', 'close', 'macd_val', 'macd_signal_line', 'rsi', 'val_is_high',
                                     'val_is_low', 'macd_is_high', 'macd_is_low', 'close_went_up', 'close_went_down'])
    raw_list = deleter(raw_list)

    for i in range(1, len(raw_list) - window_height - 1):
        buffer = list()
        for j in range(WINDOW_HEIGHT):
            buffer.append([
                float(raw_list.iloc[i + j]['high']),
                float(raw_list.iloc[i + j]['rsi']),
                float(raw_list.iloc[i + j]['close']),
                float(raw_list.iloc[i + j]['low']),
                float(raw_list.iloc[i + j]['macd_val']),
                float(raw_list.iloc[i + j]['macd_signal_line']),
                float(raw_list.iloc[i + j]['macd_is_high']),
                float(raw_list.iloc[i + j]['macd_is_low']),
                float(raw_list.iloc[i + j]['val_is_high']),
                float(raw_list.iloc[i + j]['val_is_low']),
                float(raw_list.iloc[i + j]['close_went_up']),
                float(raw_list.iloc[i + j]['close_went_down']),
            ])

        features.append(buffer)

        labels.append([[raw_list.iloc[i + j + 1]['close_went_up'], raw_list.iloc[i + j + 1]['close_went_down']]])

    features = np.array(features)
    labels = np.array(labels)

    print(labels[0], features[0])

    return features, labels


x, y = data_converter(file_names, WINDOW_HEIGHT, WAY)
x_test, y_test, x_train, y_train = x[190000:], y[190000], x[:190000], y[:190000]
print(x[:10], y[:10], sep='\n')

'''Модель тут'''
model_2 = keras.models.Sequential([
    keras.layers.Dense(12, activation='sigmoid'),
    keras.layers.Dense(20, activation='sigmoid'),
    keras.layers.Dense(20, activation='sigmoid'),
    keras.layers.Dense(2, activation='sigmoid')
])

model_1 = keras.models.Sequential([

    keras.layers.Conv2D(filters=(WINDOW_HEIGHT - 2) * (12 - 2), kernel_size=(3, 3), padding='valid',
                        strides=(1, 1), input_shape=(WINDOW_HEIGHT, 12, 1)),

    keras.layers.Reshape((-1, (WINDOW_HEIGHT - 2) * (12 - 2))),

    keras.layers.LSTM((WINDOW_HEIGHT - 2) * (12 - 2), activation='relu'),

    keras.layers.Dense(2)
])

model_3 = keras.models.Sequential([
    keras.layers.SimpleRNN(12, activation='sigmoid'),

    keras.layers.Dense(2)
])

chosen_model = model_2

chosen_model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

chosen_model.fit(x, y, epochs=EPOCHS)

chosen_model.save('main_model.keras')

prediction = chosen_model.predict(x)
print(prediction[0])

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
