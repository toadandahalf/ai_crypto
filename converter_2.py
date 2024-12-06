from constants_here import PERIODS_RSI, PERIOD_LONG_MACD, PERIOD_SHORT_MACD, PERIOD_SIGNAL_MACD, STEP_BACK, \
    WINDOW_HEIGHT
from indicators import rsi, macd, macd_to_zero_heatmap, macd_val_to_signal_heatmap, close_trend_heatmap
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import random


def data_converter_2(csv_names, window_height, way):
    '''из первого датасета берет с конца данные размера окна'''

    raw_list = []
    features = []
    labels = [0]

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

    features = []
    steps_left = len(raw_list)
    deleter = 0

    while steps_left >= 1:
        plus = random.randint(5, 15)
        minus = random.randint(7, 17)

        # Убедимся, что мы не выходим за пределы списка
        if steps_left < (plus + minus):
            break

        # Извлечение элементов из raw_list
        a = [raw_list.iloc[i] for i in range(plus)]
        features.extend(a)
        deleter += plus
        raw_list = raw_list.drop(index=[deleter + i for i in range(plus)])
        print(raw_list[:5])

        b = [raw_list.iloc[i] for i in range(minus)]
        features.extend(b)
        deleter += minus
        raw_list = raw_list.drop(index=[deleter + i for i in range(minus)])
        print(raw_list[:5])

        # Обновление labels на основе c
        c = a + b
        print(c)
        for value in c:
            print(value)
            if labels[-1] > value:
                labels.append(1)
            else:
                labels.append(0)

        steps_left -= (plus + minus)

    labels.pop(0)

    features = np.array(features)
    labels = np.array(labels)

    return [features, labels]
