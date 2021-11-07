# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Mengtong Hu mengtong@umich.edu

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ## Qestion 0 - Topics in Pandas

# ### Windowing Operations

# - an operation that perfroms an aggregation over a sliding
#   partition of values on Series or DataFrame, similar to `groubby`.

# ### Windowing Operations

# - Specify the window=n argument in `.rolling()` for the window size. 
# - After specifiying the window size, apply the appropriate
#   statistical function on top of it. Examples of statistical
#   functions include: `.sum()`, `.mean()`, `.median()`, `.var()`, `.corr()`.
# - If the offest is based on a time based column such as 'window = "2D"', the correspond
#     time based index must be monotonic.
# - The example below computes the sum of 'A' for previous 2 days

df = pd.DataFrame(np.arange(10),
   index = pd.date_range('1/1/2000', periods=10),
   columns = ['A'])
df['default sum'] = df['A'].rolling(window=3).sum()
df

# ### Windowing Operations

# - The closed parameter in `.rolling()` is used to decide the inclusions
#     of the interval endpoints in rolling window 
#     - 'right' close right endpoint
#     - 'left' close left endpoint
#     - 'both' close both endpoints
#     - 'neither' open endpoints

offset = '2D'
df["right"] = df.rolling(offset, closed="right").A.sum()  # default
df["both"] = df.rolling(offset, closed="both").A.sum()
df["left"] = df.rolling(offset, closed="left").A.sum()
df["neither"] = df.rolling(offset, closed="neither").A.sum()
df

# ### Windowing Operations
# - `.apply()` function takes an extra func argument and performs self-defined rolling computations.
