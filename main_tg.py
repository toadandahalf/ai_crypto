from telegram.ext import Application, CommandHandler, ConversationHandler
from telegram import ReplyKeyboardMarkup

from constants_here import BOT_TOKEN, SYMBOL

import time
import datetime as dt

from minute_data import get_minute_data
from prediction import get_prediction

import keras


async def start(update, context):
    symbol = f'{SYMBOL}USDT'
    buffer = []
    model = keras.models.load_model('main_model.keras')

    await update.message.reply_text('Бот начал работу, заполнение буфера',
                                    reply_markup=ReplyKeyboardMarkup([['/stop']]))
    while len(buffer) < 14:
        '''Цикл подготовки буфера'''

        now_is_s = dt.datetime.now().second

        if now_is_s == 59:
            buffer.append(get_minute_data(symbol))
            await update.message.reply_text(f'buffer: {len(buffer)}/14, time: {dt.datetime.now()}',
                                            reply_markup=ReplyKeyboardMarkup([['/stop']]))
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
            answer = get_prediction(buffer, model)
            await update.message.reply_text(f'{"ВВЕРХ" if answer == "up" else "ВНИЗ"}'
                                            f' на минуте {int(now_is_m % 60 + 1)},'
                                            f' символ: {symbol}', reply_markup=ReplyKeyboardMarkup([['/stop']]))
            time.sleep(50)

        time.sleep(0.5)


async def stop(update, context):
    await update.message.reply_text("Бот остановлен, буфер будет очищен",
                                    reply_markup=ReplyKeyboardMarkup([['/start']]))
    return ConversationHandler.END


def main():

    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("stop", stop))
    application.add_handler(CommandHandler("start", start))

    application.run_polling()


if __name__ == '__main__':
    main()
