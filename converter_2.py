from constants_here import PERIODS_RSI, PERIOD_LONG_MACD, PERIOD_SHORT_MACD, PERIOD_SIGNAL_MACD, STEP_BACK, \
    WINDOW_HEIGHT, WAY
from indicators import rsi, macd, macd_to_zero_heatmap, macd_val_to_signal_heatmap, close_trend_heatmap
import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import random
import os


def data_converter_2(csv_names, window_height, way):
    '''из первого датасета берет с конца данные размера окна'''

    raw_list = []

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
    labels = []

    reverse_or_not = 0
    flag = False
    buffer_x = []
    for i in range((len(raw_list) // 2) - 1):
        print(i)
        if reverse_or_not > 0:
            buffer_x.append(raw_list.iloc[i])
            reverse_or_not -= 1

        elif reverse_or_not == 0:
            if flag:
                reverse_or_not = random.randint(5, 17)
                flag = not flag

                features.extend(buffer_x)

                buffer_x.append(raw_list.iloc[i])
                reverse_or_not -= 1

            else:
                reverse_or_not = random.randint(5, 15) * -1
                flag = not flag

                features.extend(buffer_x[::-1])

                buffer_x.append(raw_list.iloc[i])
                reverse_or_not += 1

        else:
            buffer_x.append(raw_list.iloc[i])
            reverse_or_not += 1
    for i in range((len(raw_list) // 2) - 1):
        print(i)
        labels.append(float(1) if raw_list.iloc[i + 1]['close_went_up'] >
                                  raw_list.iloc[i + 1]['close_went_down'] else float(0))
    print('lists ready')
    print(features[:10], labels[:10])

    a, b = 0, 0
    for i in labels:
        if i == 1.0:
            a += 1
        else:
            b += 1
    print(a, b, 'plus', 'minus')

    return features, labels


file_names = os.listdir(WAY)

if file_names[0] == '.ipynb_checkpoints':
    file_names.pop(0)

x, y = data_converter_2(file_names, WINDOW_HEIGHT, WAY)
