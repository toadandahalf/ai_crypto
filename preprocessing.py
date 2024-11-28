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
    df = close_trend_heatmap(df)
    df = macd_val_to_signal_heatmap(df, 'macd_val', 'macd_signal_line')
    df = macd_to_zero_heatmap(df, 'macd_val')
    print(df)
    scaler = joblib.load('minmax_scaler.pkl')
    # scaler.transform()


get_preprocessing([[1, 2, 3],
                   [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 9, 3], [1, 2, 3], [1, 2, 3],
                   [1, 2, 3], [1, 2, 3], [1, 4, 3], [1, 2, 3],
                   [2, 2, 3], [1, 2, 3], [1, 2, 3], [1, 8, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3],
                   [1, 2, 3], [1, 2, 3], [1, 2, 3]])
