import numpy as np 
from sklearn.linear_model import LinearRegression
import pandas as pd 

train = pd.read_csv('train.csv', index_col = False)

train_columns = train.columns

target_name = train_columns[-1]


target = train[target_name].values

# target.reshape([-1,1])
x = train.drop(columns = [target_name, 'timestamp'])
print(x.values)
model = LinearRegression().fit(x,target)

r_sq = model.predict(x)

print(r_sq)

