import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from indicators import macd, rsi, macd_to_zero_heatmap, macd_val_to_signal_heatmap, close_trend_heatmap, target_ARIMA
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def data_converter_ARIMA(way,
                         window_height=1,
                         period_long_macd=26,
                         period_short_macd=12,
                         period_signal_macd=9,
                         periods_rsi=14):
    """Основная функция обработки данных"""
    raw_list = []

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
    raw_dataframe = pd.DataFrame(raw_list, columns=['timestamp', 'symbol', 'period', 'open', 'high', 'low', 'close'])
    del raw_dataframe['symbol']
    del raw_dataframe['period']
    del raw_dataframe['open']

    """Обработка данных, добавление индикаторов"""
    raw_dataframe = macd(raw_dataframe, period_long=period_long_macd,
                         period_short=period_short_macd, period_signal=period_signal_macd)
    raw_dataframe = rsi(raw_dataframe, periods=periods_rsi)
    raw_dataframe = macd_val_to_signal_heatmap(raw_dataframe, 'macd_val', 'macd_signal_line')
    raw_dataframe = macd_to_zero_heatmap(raw_dataframe, 'macd_val')
    raw_dataframe = close_trend_heatmap(raw_dataframe)
    raw_dataframe = target_ARIMA(raw_dataframe)

    """Очистка от пустых значений"""
    raw_dataframe = raw_dataframe.dropna()

    print(raw_dataframe[:4])

    """Обработка скейлером, превращение в Нампи массив"""
    scaler = MinMaxScaler()
    scaler.fit(raw_dataframe)
    prepared_dataset = scaler.transform(raw_dataframe)

    prepared_dataset = pd.DataFrame(prepared_dataset,
                                    columns=['timestamp', 'high', 'low', 'close',
                                             'macd_val', 'macd_signal_line', 'rsi',
                                             'val_is_high', 'val_is_low', 'macd_is_high', 'macd_is_low',
                                             'close_went_up', 'close_went_down',
                                             'target'])

    # del prepared_dataset['timestamp']
    prepared_dataset['timestamp'] = raw_dataframe['timestamp'].copy(deep=True)

    print(prepared_dataset[:4])

    return prepared_dataset
