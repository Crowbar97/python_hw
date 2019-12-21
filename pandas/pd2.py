from functools import reduce
import numpy as np
import pandas as pd

# Series description
s = pd.Series(np.random.randint(0, 10, 10))
print(s)
print(s.describe())

# Unique count
s = pd.Series([ 1, 2, 1, 3, 4, 3 ])
print(s.value_counts())

# Replace all Series elements to "Other"
# except two most frequent
s = pd.Series(np.random.randint(0, 10, 10))
print(s.value_counts())
s[~s.isin(s.value_counts().index[:2])] = 'Other'
print(s)

# Random with indices of 2019 days
# Sum of all Tuesdays
# Mean for each month
ds = pd.date_range(start='2019-01-01', end='2019-01-10') 
s = pd.Series(np.random.randint(0, 10, len(ds)), index=ds)
print(s)
 
tue_sum = s[s.index.weekday == 2].sum()
print(tue_sum)
 
month_mean = s.resample('M').mean()
print(month_mean)

# Reshape Series to DataFrame
s = pd.Series(np.random.randint(0, 10, 12))
df = pd.DataFrame(s.values.reshape((3, 4)))
print(df)

# Find indices divisible by 3
s = pd.Series(np.random.randint(0, 10, 12))
print(s)
print(s[s % 3 == 0].index)

# Get values by indices
s = pd.Series([ 3, 4, 5, 6, 7 ])
inds = [ 1, 3 ]
print(s[inds])

# Merging two Series vertically and horizontally
s1 = pd.Series(range(5))
s2 = pd.Series(list('abcde'))
 
print(s1.append(s2))
print(pd.concat([s1, s2], axis=1))

# Get indices of values in A that contains in B
s1 = pd.Series([ 3, 4, 5, 6, 7 ])
s2 = pd.Series([ 4, 7, 6 ])
print(np.argwhere(s1.isin(s2)).flatten())

# Get unique elements
s1 = pd.Series([ 3, 4, 3, 6, 4 ])
print(pd.Series(s.unique()))

# Upper case and char count
s = pd.Series([ 'hello', 'beautiful', 'world!' ])
print(pd.Series(word.upper() for word in s))
print(reduce(lambda accum, s2: accum + len(s2), s, 0))

# To string
s = pd.Series([ 3, 4, 5])
print(pd.Series(str(d) for d in s))

# Shift diff
s = pd.Series([ 3, 4, 5, 6, 7 ])
n = 2
print(s.diff(periods=n))

# String to date
s = pd.Series([ '2019/01/01',
                '2019-01-01',
                '01 Jan 2019' ])
print(pd.to_datetime(s))

# String date parsing
from dateutil.parser import parse
 
s = pd.Series([ '01 Feb 2019',
                '2014/04/04',
                '02-02-2011',
                '20120303',
                '2019-12-31'])
 
s_ts = s.map(lambda x: parse(x, yearfirst=True))
print(s_ts.dt.year)
print(s_ts.dt.month)
print(s_ts.dt.day)
print(s_ts.dt.weekofyear)
print(s_ts.dt.dayofyear)

# Filter with >= 2 vowels
s = pd.Series([ 'one',
                'two',
                'three',
                'four',
                'five',
                'six',
                'seven' ])
s = pd.Series(pd.Series(list(x)) for x in s)
# print(s)

vs = pd.Series(list('aeiouy'))
# print(vs)

s1 = pd.Series(filter(lambda word: sum(word.isin(vs)) > 1, s)).values
print(s1)

# Email validation
import re
 
emails = pd.Series([ 'test text @test.com',
                     'test@google.com',
                     'test.su',
                     'test@pp' ])

pattern = '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'

mask = emails.map(lambda x: bool(re.match(pattern, x)))
print(emails[mask])

