import time
import datetime as dt

from minute_data import get_minute_data
from prediction import get_prediction

symbol = 'SOLUSDT'

buffer = []

while len(buffer) < 14:
    '''Цикл подготовки буфера'''

    now_is = dt.datetime.now().second

    if now_is == 59:
        buffer.append(get_minute_data(symbol))
        print(f'buffer: {len(buffer)}/14, time: {dt.datetime.now()}')
        time.sleep(50)
    time.sleep(0.5)

action_flag_1 = False
while True:
    '''Основной цикл'''

    now_is = dt.datetime.now().second

    if now_is == 59:
        buffer.append(get_minute_data(symbol))
        buffer.pop(0)
        print(f'buffer: {len(buffer)}/14, time: {dt.datetime.now()}')
        action_flag_1 = True

    if action_flag_1:
        action_flag_1 = False
        answer = get_prediction(buffer)
        print(answer)
        time.sleep(50)

    time.sleep(0.5)
