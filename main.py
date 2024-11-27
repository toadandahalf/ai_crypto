import time
import datetime as dt

from minute_data import get_minute_data
from prediction import get_prediction

symbol = 'SOLUSDT'

time.sleep(60 - dt.datetime.now().second - 1)
previous_list = get_minute_data(symbol)

fresh_list = []
action_flag = False
time.sleep(60 - dt.datetime.now().second - 1)
while True:
    '''Основной цикл'''

    now_is = dt.datetime.now().second

    if now_is == 58:
        fresh_list = get_minute_data(symbol)
        action_flag = True

    if now_is == 59:
        fresh_list = get_minute_data(symbol)
        action_flag = True

    if action_flag:
        action_flag = False
        answer = get_prediction(fresh_list, previous_list)
        previous_list = fresh_list
        print(answer)
