import pandas as pd
import numpy as np
import pprint as pp

df = pd.DataFrame({'normal': [1.0, 2.0, 3.0, 4.0, 5.0]})
df['shifted'] = df.shift(1)

print(df)

a=[]
a.append(1)
a.append(2)
print(a)

print(int(123 % 60))