import pandas as pd
import numpy as np
import numpy.random as nprand
import math

df = pd.read_csv("data/train.csv")
dfn = df.to_numpy()
dfn = np.hsplit(dfn, [1])
y = dfn[0]
x = dfn[1]
