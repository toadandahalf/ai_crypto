import pandas as pd
import random


def deleter(dataframe):
    print(count := (dataframe[dataframe['close_went_down'] == 1]).count()['high'] - (dataframe[dataframe['close_went_up'] == 1]).count()['high'])
    for i in range(count):
        print(f'{i + 1}/{count} {len(dataframe)}')
        random_one = dataframe[dataframe['close_went_down'] == 1].sample().index
        dataframe.drop(random_one, inplace=True)

    return dataframe
