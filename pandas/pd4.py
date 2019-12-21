import sys
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

dataset_url = 'https://raw.githubusercontent.com' \
              '/Grossmend/CSV/master/titanic/data.csv'

# Read csv to DataFrame
tit_df = pd.read_csv(dataset_url)
print(tit_df.columns)
print(tit_df.head())

# Read specified cols (Name, Sex, Survived)
# and rows of csv to DataFrame
row_count = 5
tit_df = pd.read_csv(dataset_url,
                     nrows=row_count,
                     usecols=[ 'Name', 'Sex', 'Survived' ])
print(tit_df.columns)
print(tit_df.head())

# Read csv and append each 100th row to DataFrame
tf_reader = pd.read_csv(dataset_url, chunksize=100)
df = pd.DataFrame()
for chunk in tf_reader:
    df = df.append(chunk.iloc[0,:])
print(df.head())

# Series indices to DataFrame column
s = pd.Series([ 'Hello', 'darkness', 'my', 'old', 'friend!' ])
print(s)
 
df = s.to_frame().reset_index()
print(df)

df.columns = [ 'index', 'number' ]
print(df)

# Get DataFrame information
df = pd.read_csv(dataset_url)
print(df.shape)
print(df.dtypes)
print(df.describe())

