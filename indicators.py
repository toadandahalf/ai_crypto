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


def target_ARIMA(data):
    data['target'] = (data['close'] < data['close'].shift(-1)).astype(int)

    return data
