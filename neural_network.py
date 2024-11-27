import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
import pprint as pp
import tensorflow as tf
import os

EPOCHS = 20
WINDOW_HEIGHT = 1
PERIODS_RSI = 7
PERIOD_LONG_MACD = 13
PERIOD_SHORT_MACD = 6
PERIOD_SIGNAL_MACD = 9
STEP_BACK = max(PERIOD_SIGNAL_MACD, PERIOD_SHORT_MACD, PERIOD_LONG_MACD, PERIODS_RSI, WINDOW_HEIGHT)
WAY = 'raw_data_SOL'

file_names = os.listdir(WAY)

if file_names[0] == '.ipynb_checkpoints':
    file_names.pop(0)

pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)

# source = 'https://public.bybit.com'


"""
Exponential moving average
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
Params: 
    data: pandas DataFrame
    period: smoothing period
    column: the name of the column with values for calculating EMA in the 'data' DataFrame
    
Returns:
    copy of 'data' DataFrame with 'ema[period]' column added
"""


def ema(data, period=0, column='close'):
    data['ema' + str(period)] = data[column].ewm(ignore_na=False, min_periods=period, com=period, adjust=True).mean()

    return data


"""
Moving Average Convergence/Divergence Oscillator (MACD)
Source: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
Params: 
    data: pandas DataFrame
    period_long: the longer period EMA (26 days recommended)
    period_short: the shorter period EMA (12 days recommended)
    period_signal: signal line EMA (9 days recommended)
    column: the name of the column with values for calculating MACD in the 'data' DataFrame
    
Returns:
    copy of 'data' DataFrame with 'macd_val' and 'macd_signal_line' columns added
"""


def macd(data, period_long=26, period_short=12, period_signal=9, column='close'):
    remove_cols = []
    if not 'ema' + str(period_long) in data.columns:
        data = ema(data, period_long)
        remove_cols.append('ema' + str(period_long))

    if not 'ema' + str(period_short) in data.columns:
        data = ema(data, period_short)
        remove_cols.append('ema' + str(period_short))

    data['macd_val'] = data['ema' + str(period_short)] - data['ema' + str(period_long)]
    data['macd_signal_line'] = data['macd_val'].ewm(ignore_na=False, min_periods=0, com=period_signal,
                                                    adjust=True).mean()

    data = data.drop(remove_cols, axis=1)

    return data


def rsi(data, periods=14, close_col='close'):
    data['rsi_u'] = 0.
    data['rsi_d'] = 0.

    for index in range(1, len(data)):
        change = float(data.at[index, close_col]) - float(data.at[index - 1, close_col])
        if change > 0:
            data.at[index, 'rsi_u'] = change
        else:
            data.at[index, 'rsi_d'] = -change

    # Рассчитываем средние значения
    avg_gain = data['rsi_u'].rolling(window=periods).mean()
    avg_loss = data['rsi_d'].rolling(window=periods).mean()

    # Избегаем деления на ноль
    rs = avg_gain / avg_loss.replace(0, float('nan'))  # заменяем нули на NaN
    data['rsi'] = 100 - (100 / (1 + rs))

    return data.drop(['rsi_u', 'rsi_d'], axis=1)


def close_trend_heatmap(data):
    data['close_went_up'] = (data['close'] > data['close'].shift(1)).astype(int)
    data['close_went_down'] = (data['close'] <= data['close'].shift(1)).astype(int)

    return data


def macd_val_to_signal_heatmap(data, macd_val, macd_signal_line):
    data['val_is_high'] = (data[macd_val] > data[macd_signal_line]).astype(int)
    data['val_is_low'] = (data[macd_val] <= data[macd_signal_line]).astype(int)

    return data


def macd_to_zero_heatmap(data, macd_val):
    data['macd_is_high'] = (data[macd_val] > 0).astype(int)
    data['macd_is_low'] = (data[macd_val] <= 0).astype(int)

    return data


def data_converter(csv_names, WINDOW_HEIGHT, way):
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
    raw_list = close_trend_heatmap(raw_list)
    raw_list = macd_val_to_signal_heatmap(raw_list, 'macd_val', 'macd_signal_line')
    raw_list = macd_to_zero_heatmap(raw_list, 'macd_val')
    raw_list = raw_list.dropna()

    scaler = MinMaxScaler()
    scaler.fit(raw_list)
    raw_list = scaler.transform(raw_list)
    raw_list = pd.DataFrame(raw_list,
                            columns=['high', 'low', 'close', 'macd_val', 'macd_signal_line', 'rsi', 'val_is_low',
                                     'val_is_high', 'close_went_down', 'close_went_up', 'macd_is_low', 'macd_is_high'])

    for i in range(1, len(raw_list) - WINDOW_HEIGHT):
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
                float(raw_list.iloc[i + j]['val_is_low'])
            ])

        labels.append([
            float(raw_list.iloc[i + j + 1]['close_went_up']),
            float(raw_list.iloc[i + j + 1]['close_went_down'])
        ])

        features.append(buffer)

    features = np.array(features)
    labels = np.array(labels)

    return [features, labels]


converted_data = data_converter(file_names, WINDOW_HEIGHT, WAY)

x = converted_data[0]
y = converted_data[1]
WINDOW_LENGHT = len(x[0][0])

'''model_1 = keras.models.Sequential([

    keras.layers.Conv2D(filters=(WINDOW_HEIGHT - 2) * (WINDOW_LENGHT - 2), kernel_size=(3, 3), padding='valid',
strides=(1, 1), input_shape=(WINDOW_HEIGHT, WINDOW_LENGHT, 1)),

    keras.layers.Reshape((-1, (WINDOW_HEIGHT - 2) * (WINDOW_LENGHT - 2))),

    keras.layers.LSTM((WINDOW_HEIGHT - 2) * (WINDOW_LENGHT - 2), activation='relu'),

    keras.layers.Dense(1)
])'''

model_2 = keras.models.Sequential([
    keras.layers.LSTM(WINDOW_LENGHT, activation='sigmoid'),

    keras.layers.Dense(2, activation='sigmoid')
])

chosen_model = model_2

chosen_model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

chosen_model.fit(x, y, epochs=10)

chosen_model.save('main_model.keras')
