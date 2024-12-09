import joblib
from indicators import macd, rsi, macd_to_zero_heatmap, macd_val_to_signal_heatmap
from constants_here import PERIODS_RSI, PERIOD_SIGNAL_MACD, PERIOD_SHORT_MACD, PERIOD_LONG_MACD, WINDOW_HEIGHT
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def data_converter_3(window_height):
    features = []
    labels = []

    shortened_list = joblib.load('shortened_list_SOL_500')
    shortened_list = macd(shortened_list, period_long=PERIOD_LONG_MACD, period_short=PERIOD_SHORT_MACD,
                          period_signal=PERIOD_SIGNAL_MACD)
    shortened_list = rsi(shortened_list, PERIODS_RSI)
    shortened_list = macd_val_to_signal_heatmap(shortened_list, 'macd_val', 'macd_signal_line')
    shortened_list = macd_to_zero_heatmap(shortened_list, 'macd_val')
    shortened_list = shortened_list.dropna()

    scaler = MinMaxScaler()
    scaler.fit(shortened_list)
    shortened_list = scaler.transform(shortened_list)
    joblib.dump(scaler, 'minmax_scaler.pkl')

    shortened_list = pd.DataFrame(shortened_list,
                                  columns=['high', 'low', 'close', 'close_went_up', 'close_went_down', 'macd_val',
                                           'macd_signal_line',
                                           'rsi', 'val_is_high', 'val_is_low', 'macd_is_high',
                                           'macd_is_low'])

    for i in range(1, len(shortened_list) - window_height - 1):
        buffer = list()
        buffer.append([
            float(shortened_list.iloc[i]['high']),
            float(shortened_list.iloc[i]['rsi']),
            float(shortened_list.iloc[i]['close']),
            float(shortened_list.iloc[i]['low']),
            float(shortened_list.iloc[i]['macd_val']),
            float(shortened_list.iloc[i]['macd_signal_line']),
            float(shortened_list.iloc[i]['macd_is_high']),
            float(shortened_list.iloc[i]['macd_is_low']),
            float(shortened_list.iloc[i]['val_is_high']),
            float(shortened_list.iloc[i]['val_is_low']),
            float(shortened_list.iloc[i]['close_went_up']),
            float(shortened_list.iloc[i]['close_went_down']),
        ])

        features.append(buffer)

        labels.append([[shortened_list.iloc[i + 1]['close_went_up'], shortened_list.iloc[i + 1]['close_went_down']]])

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

# x, y = data_converter_3(WINDOW_HEIGHT)
# print(x[:10], y[:10], sep='\n')
