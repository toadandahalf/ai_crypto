import time
import datetime as dt
from constants_here import BOT_TOKEN, USERS
from telegram import Bot

from minute_data import get_minute_data
from prediction import get_prediction

bot = Bot(token=BOT_TOKEN)
for user_id in USERS:
    bot.send_message(chat_id=user_id, text='Бот начал работу')

symbol = 'SOLUSDT'

buffer = []

while len(buffer) < 14:
    '''Цикл подготовки буфера'''

    now_is_s = dt.datetime.now().second

    if now_is_s == 59:
        buffer.append(get_minute_data(symbol))
        print(f'buffer: {len(buffer)}/14, time: {dt.datetime.now()}')
        time.sleep(50)
    time.sleep(0.5)

action_flag_1 = False
while True:
    '''Основной цикл'''

    now_is_s = dt.datetime.now().second
    now_is_m = dt.datetime.now().minute

    if now_is_s == 59:
        buffer.append(get_minute_data(symbol))
        buffer.pop(0)
        print(f'buffer: {len(buffer)}/14, time: {dt.datetime.now()}')
        action_flag_1 = True

    if action_flag_1:
        action_flag_1 = False
        answer = get_prediction(buffer)

        for user_id in USERS:
            bot.send_message(chat_id=user_id, text=f'{"↑" if answer == "up" else "↓/→"} на минуту:'
                                                   f' {int(now_is_m % 60)}, символ: {symbol}')
        time.sleep(50)

    time.sleep(0.5)
