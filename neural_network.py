import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
import os
import joblib
from indicators import macd, rsi, close_trend_heatmap, macd_to_zero_heatmap, macd_val_to_signal_heatmap

EPOCHS = 20
WINDOW_HEIGHT = 1
PERIODS_RSI = 7
PERIOD_LONG_MACD = 13
PERIOD_SHORT_MACD = 6
PERIOD_SIGNAL_MACD = 9
STEP_BACK = max(PERIOD_SIGNAL_MACD, PERIOD_SHORT_MACD, PERIOD_LONG_MACD, PERIODS_RSI, WINDOW_HEIGHT)
WAY = 'raw_data_LUNA'

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

    for i in range(1, len(raw_list) - window_height):
        buffer = []
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

        labels.append([
            float(raw_list.iloc[i + j + 1]['close_went_up']),
            float(raw_list.iloc[i + j + 1]['close_went_down'])
        ])

        features.append(buffer)

    features = np.array(features)
    labels = np.array(labels)

    print(labels[0], features[0])

    return [features, labels]


converted_data = data_converter(file_names, WINDOW_HEIGHT, WAY)

x = converted_data[0]
y = converted_data[1]
WINDOW_LENGTH = len(x[0][0])

model_2 = keras.models.Sequential([
    keras.layers.LSTM(12, activation='sigmoid'),
    keras.layers.Dense(2, activation='sigmoid')
])

model_2.compile(optimizer='adam', loss='binary_crossentropy')

chosen_model = model_2

chosen_model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

chosen_model.fit(x, y, epochs=10)

chosen_model.save('main_model.keras')

print(x[0])