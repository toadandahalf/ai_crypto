import requests
# import datetime as dt


def get_minute_data(symbol):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=1'

    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверка на ошибки HTTP
        data = response.json()

        if data:
            last_candle = data[-1]
            # open_time = dt.datetime.fromtimestamp(last_candle[0] / 1000)  # Время открытия свечи
            # open_price = float(last_candle[1])  # Цена открытия
            high_price = float(last_candle[2])  # Максимальная цена
            low_price = float(last_candle[3])  # Минимальная цена
            close_price = float(last_candle[4])  # Цена закрытия

            return [high_price, low_price, close_price]
        else:
            print("Нет данных")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе данных: {e}")
