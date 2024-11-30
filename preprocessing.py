import random

import pandas
import numpy
from sklearn.preprocessing import StandardScaler
import joblib
from indicators import macd, rsi, close_trend_heatmap, macd_val_to_signal_heatmap, macd_to_zero_heatmap

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

PERIODS_RSI = 7
PERIOD_LONG_MACD = 13
PERIOD_SHORT_MACD = 6
PERIOD_SIGNAL_MACD = 9


def get_preprocessing(data):
    df = pandas.DataFrame([*data], columns=['high', 'low', 'close'])
    df = macd(df, period_long=PERIOD_LONG_MACD, period_short=PERIOD_SHORT_MACD,
              period_signal=PERIOD_SIGNAL_MACD)
    df = rsi(df, periods=PERIODS_RSI)
    df = macd_val_to_signal_heatmap(df, 'macd_val', 'macd_signal_line')
    df = macd_to_zero_heatmap(df, 'macd_val')
    df = close_trend_heatmap(df)
    df = df.dropna()

    scaler = joblib.load('minmax_scaler.pkl')
    df = scaler.transform(df)
    df = pandas.DataFrame(df,
                          columns=['high', 'low', 'close', 'macd_val', 'macd_signal_line', 'rsi', 'val_is_low',
                                   'val_is_high', 'macd_is_low', 'macd_is_high', 'close_went_up', 'close_went_down'])

    return df.iloc[-1].values.reshape(1, 1, -1)


# print(get_preprocessing([[random.randint(100, 150) for i in range(3)] for ii in range(14)]))
