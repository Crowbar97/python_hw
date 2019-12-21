# Module importing
import sys
import numpy as np
import pandas as pd

# Create Series from list, dict and np object
lst = [ 4, 5, 6 ]
dct = { "one" : 1,
        "two": 2 }
np_arr = np.zeros((3))

print(pd.Series(lst))
print(pd.Series(dct))
print(pd.Series(np_arr))

# Convert Series to DataFrame
s = pd.Series({ 1: "one",
                2: "two",
                3: "three" })
df = s.to_frame()
print(df)

# Integrate multiple Series into one DataFrame
s1 = pd.Series([ 1, 2, 3 ])
s2 = pd.Series([ 4, 5, 6]) 
df = pd.concat([ s1, s2 ], axis=1)
print(df)

# Assign name to Series
s1 = pd.Series([ 1, 2, 3 ])
s1.name = "my series"
print(s1)

# Get A \ B
s1 = pd.Series([ 1, 2, 3, 4, 5 ])
s2 = pd.Series([ 1, 3, 5]) 
ocs = s1.isin(s2)
sub = s1[~ocs]
print(sub)
print(np.setdiff1d(s1, s2))

# Get unique values from A and B
s1 = pd.Series([ 1, 2, 3, 4, 5 ])
s2 = pd.Series([ 1, 3, 5, 7 ]) 

union = pd.Series(np.union1d(s1, s2))
intersect = pd.Series(np.intersect1d(s1, s2))
unique = union[~union.isin(intersect)]
print(unique)
 
unique = np.setxor1d(s1, s2)
print(unique)

