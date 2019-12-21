import numpy as np
import pandas as pd

# Euclidean distance
n = 3
s1 = pd.Series(np.random.randint(0, 10, n))
s2 = pd.Series(np.random.randint(0, 10, n))
 
print(np.linalg.norm(s1 - s2))

# Series local maximum indices
s = pd.Series([ 1, 5, 7, 11, 8, 4, 12, 5, 8, 16, 8 ])
d1 = np.sign(np.diff(s))
print(d1)
d2 = np.diff(d1)
print(d2)
print(np.where(d2 == -2)[0] + 1)

# Replace spaces with least frequent char
str1 = "Hello darkness my old friend!"
s = pd.Series(list(str1))
fs = s.value_counts()
print(fs)
lf = fs.index[-1]
print(str1.replace(' ', lf))

# Random Saturday Series
s = pd.Series(np.random.randint(0, 10, 10),
              pd.date_range('2018-01-01',
                            periods=10,
                            freq='W-SAT'))
print(s)

# Fill missing with values above
s = pd.Series([ 2, 5, 8, np.nan ],
              index=pd.to_datetime([ '2018-01-01',
                                     '2018-01-03',
                                     '2018-01-06',
                                     '2018-01-08' ]))
print(s.resample('D').ffill())

# Correlation
s = pd.Series(np.random.randint(0, 10, 10))
print(s.autocorr())


