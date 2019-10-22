import numpy as np 
import pandas as pd 


buildings = pd.read_csv('building_metadata.csv')

usage = buildings['primary_use']

usage = usage.values

usage = set(usage)

usage = list(usage)

index = [x for x in range(len(usage))]

# print(index)
x = {'primary_usage' : usage, 'index':index}
df = pd.DataFrame(x)

df.to_csv('primary_usage_translations.csv')

