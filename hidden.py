import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from data_converter_ARIMA import data_converter_ARIMA
from constants_here import WAY, PERIODS_RSI, PERIOD_SIGNAL_MACD, PERIOD_SHORT_MACD, PERIOD_LONG_MACD, WINDOW_HEIGHT

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Загрузка данных
data = data_converter_ARIMA(WAY, WINDOW_HEIGHT, PERIOD_LONG_MACD, PERIOD_SHORT_MACD, PERIOD_SIGNAL_MACD, PERIODS_RSI)
# data['timestamp'] = pd.to_datetime(data['timestamp'])
# data.set_index('timestamp', inplace=True)

# Проверка на стационарность
result = adfuller(data['target'])  # Замените 'Value' на вашу целевую колонку
if result[1] > 0.05:
    data['target'] = data['target'].diff().dropna()  # Применение первой разности

# Определение параметров p и q с помощью ACF и PACF (графики должны быть построены здесь)

# Создание модели ARIMA
model = ARIMA(data['target'], order=(5, 4, 3))  # Замените p, d, q на выбранные параметры
model_fit = model.fit()

# Прогнозирование
forecast = model_fit.forecast(steps=1)  # Прогноз на 10 шагов вперед

# Визуализация результатов
plt.plot(data.index[-100:], data['target'][-100:], label='Исторические данные')
plt.plot(pd.date_range(start=data.index[-1], periods=11)[1:], forecast, label='Прогноз', color='red')
plt.legend()
plt.show()