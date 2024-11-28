import time
import datetime as dt

from minute_data import get_minute_data
from prediction import get_prediction

symbol = 'SOLUSDT'

buffer = []
for i in range(14):
    time.sleep(60 - dt.datetime.now().second - 1)
    buffer.append(get_minute_data(symbol))
time.sleep(60 - dt.datetime.now().second - 1)

action_flag = False
while True:
    '''Основной цикл'''

    now_is = dt.datetime.now().second

    if now_is == 59:
        buffer.append(get_minute_data(symbol))
        buffer.pop(0)
        action_flag = True

    if action_flag:
        action_flag = False
        answer = get_prediction(buffer)
        print(answer)
    time.sleep(0.5)
