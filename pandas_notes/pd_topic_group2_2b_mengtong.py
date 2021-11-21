# -*- coding: utf-8 -*-
# ## Topics in Pandas
# **Stats 507, Fall 2021**
#

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [If-Then](#If-Then)
# + [Time Delta](#Time-Delta)
# + [Sorting](#Sorting)
# + [Timestamp class](#Timestamp-class)
# + [Table Styler](#Table-Styler)
# + [Pandas.concat](#Pandas.concat)
# + [Windowing Operations](#Windowing-Operations)

# ## If Then
# **Kailin Wang**
# **wkailin@umich.edu**

# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
from os.path import exists

# ## Pandas `if-then`  idioms
# - The `if-then/if-then-else` idiom is a compact form of if-else that can be implemented to columns in `pd.DataFrame`
# - Expressed on one column, and assignment to another one or more columns
# - Use pandas where after you’ve set up a mask

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
df

# ## Pandas `if-then`  idioms
# - An `if-then` on one column

df.loc[df.AAA >= 5, "BBB"] = -1
df

# - An `if-then` with assignment to 2 columns:

df.loc[df.AAA >= 5, ["BBB", "CCC"]] = 1022
df

# ## Pandas `if-then`  idioms
# - Use pandas where after you’ve set up a mask

df_mask = pd.DataFrame(
    {"AAA": [True] * 4, "BBB": [False] * 4, "CCC": [True, False] * 2}
)
df.where(df_mask,1022)

# ## Pandas `if-then-else`  idioms
# - if-then-else using NumPy’s where()

df = pd.DataFrame(
    {"AAA": [4, 5, 6, 7], "BBB": [10, 20, 30, 40], "CCC": [100, 50, -30, -50]}
)
df
df["logic"] = np.where(df["AAA"] > 5, "high", "low")
df

# ## Time Delta
# **Liuyu Tao**
# **liuyutao@umich.edu**

# ## Overview
# - Parsing
# - to_timedelta

# ## Parsing
# - There are several different methods to construct the Timeselta, below are the examples

# +
import pandas as pd
import datetime

# read as "string"
print(pd.Timedelta("2 days 3 minutes 36 seconds"))
# similar to "datetime.timedelta"
print(pd.Timedelta(days=2, minutes=3, seconds=36))
# specify the integer and the unit of the integer
print(pd.Timedelta(2.0025, unit="d"))
# -

# ## Sorting
# **Julia Weber- juliaweb@umich.edu**

# ## Sorting- About
# - Pandas has built in functions that allow the user to sort values in a column or index of a dataframe.
# - Sorting is important, as a user can look for patterns in the data and easily determine which observations have the highest/lowest values for a certain variable.

# ## sort_values() Function
# - The sort_values() function can be used to order rows of a dataframe by the values of a column.
# - Default sorts low to high. If we set ascending=False, sorts high to low.

# +
import pandas as pd

names = ["Julia", "James", "Andrew", "Sandy", "Joe"]
ages = [15, 18, 16, 30, 26]
test_df = pd.DataFrame({"name" : names, "age" : ages})
test_df.sort_values("age", ascending=False)
# -

# ## sort_index() Function
# - The sort_index() function can be used to sort the index of a dataframe.
# - This function is similar to the sort_values() function, but is applied to the index.

sorted_df = test_df.sort_values("age", ascending=False)
sorted_df.sort_index()

# ## Timestamp class
# **Yuelin He- yuelinhe@umich.edu**
#

# Timestamp is Pandas' equivalent (and usually interchangeable) class of 
# python’s Datetime. To construct a Timestamp, there are three calling 
# conventions:
#
# 1. Converting a datetime-like string.
#
# 1. Converting a float representing a Unix epoch in units of seconds.
#
# 1. Converting an int representing a Unix-epoch in units of seconds in a 
# specified timezone.
#
# The form accepts four parameters that can be passed by position or keyword.
#
# There are also forms that mimic the API for datetime.datetime (with year, 
# month, day, etc. passed separately through parameters).
#
# See the following code for corresponding examples:

# +
import pandas as pd

## datetime-like string
print(pd.Timestamp('2021-01-01T12'))

## float, in units of seconds
print(pd.Timestamp(889088900.5, unit='s'))

##int, in units of seconds, with specified timezone
print(pd.Timestamp(5201314, unit='s', tz='US/Pacific'))
# -

# In Pandas, there are many useful attributes to do quick countings in Timestamp.
#
# - Counting the day of the...
#     + week: using *day_of_week*, *dayofweek*
#     + year: using *day_of_year*, *dayofyear*
# - Counting the week number of the year: using *week*, *weekofyear*
# - Counting the number of days in that month: using *days_in_month*, *daysinmonth*
#

# +
# Counting the day of the week
ts = pd.Timestamp(2018, 3, 21)
print(ts.day_of_week)
print(ts.dayofweek)

# Counting the day of the year
print(ts.day_of_year)
print(ts.dayofyear)

# Counting the week number of the year
print(ts.week)
print(ts.weekofyear)

# Counting the number of days in that month
print(ts.days_in_month)
print(ts.daysinmonth)
# -

# Whether certain characteristic is true can also be determined.
#
# - Deciding if the date is the start of the...
#     + year: using *is_year_start*
#     + quarter: using *is_quarter_start*
#     + month: using *is_month_start*
# - Similarly, deciding if the date is the end of the...
#     + year: using *is_year_end*
#     + quarter: using *is_quarter_end*
#     + month: using *is_month_end*
# - Deciding if the year is a leap year: using *is_leap_year*

# +
# # Start?
print(pd.Timestamp(2000, 1, 1).is_year_start)
print(pd.Timestamp(2000, 2, 1).is_quarter_start)
print(pd.Timestamp(2000, 3, 1).is_month_start)

# # End?
print(pd.Timestamp(2000, 12, 31).is_year_end)
print(pd.Timestamp(2000, 12, 30).is_quarter_end)
print(pd.Timestamp(2000, 11, 30).is_month_start)

# Leap year?
print(pd.Timestamp(2000, 12, 31).is_leap_year)
print(pd.Timestamp(2001, 12, 30).is_leap_year)
# -

#
# Reference: 
# https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html#

# ## Table Styler
# ### Manipulate many parameters of a table using the table Styler object in pandas.
# **Xiying Gao**

# ## Pandas.concat
# **Ziyin Chen- email: edwardzc@umich.edu**

# ## General Discription
# * Concatenate pandas objects along a particular axis with optional set logic along the other axes.
#
# * Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.

# ## concat
# * used to combine tow dataframe or combining two series 
#     1. can be used to join two DataFrame or Series with or without similar column with the inclusion of `join = `
#     2. can be used to join two DataFrames either vertially or horizontally with `axis = 1`
#     

# ## Example 1 
# join two dataframe horizontaly and vertially

# +
import pandas as pd 
from IPython.display import display

dic1 = {'Name': ['Allen', 'Bill','Charle','David','Ellen'],
      'number':[1,2,3,4,5],
      'letter':['a','b','c','d','e']}
dic2 = {'A':['a','a','a','a','a'],
       'B':['b','b','b','b','b'],
       'number':[10,11,12,13,14]}
df1 = pd.DataFrame(dic1)
df2 = pd.DataFrame(dic2)
display(df1)
display(df2)
# -

# join vertially 

df = pd.concat([df1,df2])
display(df)

# join horizontally 

df = pd.concat([df1,df2],axis =1 )
display(df)

# ## Example 2 
# join with the common column
#

df = pd.concat([df1,df2],join='inner')
display(df)

# ## Windowing Operations
# **Mengtong Hu- mengtong@umich.edu**

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
