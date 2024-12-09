from constants_here import WAY, STEP_BACK
import pandas as pd
import numpy as np
import os
from indicators import close_trend_heatmap
import joblib

file_names = os.listdir(WAY)

if file_names[0] == '.ipynb_checkpoints':
    file_names.pop(0)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
np.set_printoptions(threshold=np.inf)

raw_list = []

with open(f'{WAY}/{file_names[0]}', 'r') as first_to_pop:
    raw_list.extend(
        list(map(lambda x: x.rstrip().split(','), first_to_pop.readlines()[- (STEP_BACK + 1):]))
    )
file_names.pop(0)

for i in file_names:
    with open(f'{WAY}/{i}', 'r') as csv_file:
        raw_list.extend(list(map(lambda x: x.rstrip().split(','), csv_file.readlines()[1:]))
                        )

raw_list = pd.DataFrame(raw_list, columns=['timestamp', 'symbol', 'period', 'open', 'high', 'low', 'close'])
del raw_list['timestamp']
del raw_list['symbol']
del raw_list['period']
del raw_list['open']
raw_list = close_trend_heatmap(raw_list)
raw_list = raw_list.dropna()

print(raw_list[:10])

for i in range(len(raw_list)):
    a, b = (raw_list['close_went_up'] == 1).sum(), len(raw_list) // 2

    print(f'{a}/{b - a} - up/diff')
    if a + 500 <= b:
        raw_list = raw_list[:-1]
    else:
        break
print(len(raw_list))

joblib.dump(raw_list, 'shortened_list_SOL_500')