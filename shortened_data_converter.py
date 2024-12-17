import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from indicators import macd, rsi, macd_to_zero_one_hot, macd_val_to_signal_one_hot, close_trend_one_hot
import os


def shortened_data_converter(way,
                             window_height=1,
                             period_long_macd=26,
                             period_short_macd=12,
                             period_signal_macd=9,
                             periods_rsi=14,
                             until=50):
    """Функция обработки данных, содержащая модуль
     удаления строк до достижения заданной разницы
     в количестве значений ВВЕРХ и ВНИЗ в датасете"""
    raw_list = []
    features = []
    labels = []

    """Какое окно запаса надо делать, чтобы не получить IndexError"""
    step_back = max(window_height, periods_rsi, period_signal_macd, period_short_macd, period_long_macd)

    """Получение данных"""
    csv_names = os.listdir(way)

    if csv_names[0] == '.ipynb_checkpoints':
        csv_names.pop(0)

    with open(f'{way}/{csv_names[0]}', 'r') as first_to_pop:
        raw_list.extend(
            list(map(lambda x: x.rstrip().split(','), first_to_pop.readlines()[- (step_back + 1):]))
        )
    csv_names.pop(0)

    for i in csv_names:
        with open(f'{way}/{i}', 'r') as csv_file:
            raw_list.extend(list(map(lambda x: x.rstrip().split(','), csv_file.readlines()[1:]))
                            )

    """Удаление ненужных столбцов"""
    raw_list = pd.DataFrame(raw_list, columns=['timestamp', 'symbol', 'period', 'open', 'high', 'low', 'close'])
    del raw_list['timestamp']
    del raw_list['symbol']
    del raw_list['period']
    del raw_list['open']

    """Обработка данных, добавление индикаторов"""
    raw_list = macd(raw_list, period_long=period_long_macd,
                    period_short=period_short_macd, period_signal=period_signal_macd)
    raw_list = rsi(raw_list, periods=periods_rsi)
    raw_list = macd_val_to_signal_one_hot(raw_list, 'macd_val', 'macd_signal_line')
    raw_list = macd_to_zero_one_hot(raw_list, 'macd_val')
    raw_list = close_trend_one_hot(raw_list)

    """Ручная проверка"""
    print(raw_list[:10])

    for i in range(len(raw_list)):
        pluses, half = (raw_list['close_went_up'] == 1).sum(), len(raw_list) // 2

        print(f'{pluses}/{half - pluses} - up/diff')
        if pluses + until <= half:
            raw_list = raw_list[:-1]
        else:
            break

    """Проверка длины датасета на выходе"""
    print(len(raw_list))

    """Очистка от пустых значений"""
    raw_list = raw_list.dropna()

    """Обработка скейлером, превращение в Нампи массив"""
    scaler = MinMaxScaler()
    scaler.fit(raw_list)
    raw_list = scaler.transform(raw_list)

    """Превращение обратно в массив Пандас"""
    raw_list = pd.DataFrame(raw_list,
                            columns=['high', 'low', 'close', 'macd_val', 'macd_signal_line', 'rsi', 'val_is_high',
                                     'val_is_low', 'macd_is_high', 'macd_is_low', 'close_went_up', 'close_went_down'])

    """Составление сета для обучения с определенной высотой окна"""
    for i in range(1, len(raw_list) - window_height - 1):
        buffer = list()

        for j in range(window_height):
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

    """Переведение данных в Наспи массив для обучения"""
    features = np.array(features)
    labels = np.array(labels)

    return features, labels
