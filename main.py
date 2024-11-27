import time
import datetime as dt

from minute_data import get_minute_data
from prediction import get_prediction

symbol = 'SOLUSDT'

time.sleep(60 - dt.datetime.now().second - 1)
previous_list =

buffer = []
action_1 = False
time.sleep(60 - dt.datetime.now().second - 1)
while True:
    '''Основной цикл'''

    now_is = dt.datetime.now().second

    if now_is == 58:
        buffer = get_minute_data(symbol)
        action_1 = True

    if now_is == 59 and action_1:
        get_prediction(get_minute_data(symbol))
        buffer.clear()
        action_1 = False

    elif action_1:
        get_prediction(buffer)
        buffer.clear()
        action_1 = False
