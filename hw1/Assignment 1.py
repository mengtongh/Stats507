# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # STATS507 Assignment 1
# ### **Kailan Xu**  
# ### *September 14,2021*

# %% [markdown]
#   

# %% [markdown]
# ## Question 0 - Markdown warmup [10 points]                     
# ***

# %% [markdown]
#
# This is $question \ o$ for [problem set 1][ps1] of [Stats 507][s507].
#
# ```
#  Question 0 is about Markdown.
# ```
#
# The next question is about the **Fibonnaci sequence**, $Fn=Fn−2+Fn−1$. In part **a** we will define a Python function `fib_rec()`.
#
# Below is a …
#
# ### Level 3 Header
# Next, we can make a bulleted list:
#
# - Item 1
#     - detail 1
#     - detail 2
# - Item 2
#
# Finally, we can make an enumerated list:
# - Item 1
# - Item 2
#
# [ps1]:https://jbhender.github.io/Stats507/F21/ps/ps1.html
# [s507]:https://jbhender.github.io/Stats507/F21/
#
#
#

# %% [markdown]
# ***

# %%
# libraries: -----------------------------------------------------------------
import pandas as pd
import numpy as np
import time 
from itables import show
from scipy.stats import norm


# %% [markdown]
# ## Question 1 - - Fibonnaci Sequence [30]

# %% [markdown]
# #### a. Write a $recursive$ function `fib_rec()` that takes a single input `n` and returns the value of $Fn$.
#
#

# %%
# functions: -----------------------------------------------------------------
def fib_rec(n):
    """
    Compute the nth value of Fibonnaci Sequence.
    
    Parameter
    ---------
    n: int
    
    Returns
    ---------
    An integer for the nth value of Fib Seq
    
    """
    if n == 0 or n == 1:
        return n
    else:
        return fib_rec(n-1) + fib_rec(n-2)
    
# test function: -------------------------------------------------------------
def test_func(func):
    for i in [7,11,13]:
        print("F"+str(i)+"=",func(i))


# %%
# test result: ---------------------------------------------------------------
test_func(fib_rec)


# %% [markdown]
# ***

# %% [markdown]
# #### b.  Write a function `fib_for()` with the same signature that computes $Fn$  by summation using a `for` loop.
#
#

# %%
# functions: -----------------------------------------------------------------
def fib_for(n):
    if n == 0 or n == 1:
        return n
    else:
        fa = 0
        fb = 1
        fn = 1
        for i in range(2,n):
            fa = fb
            fb = fn
            fn= fa + fb
        return fn

    
    
# test result: ---------------------------------------------------------------
test_func(fib_for)


# %% [markdown]
# ***

# %% [markdown]
# **c. Write a function `fib_whl()` with the same signature that computes $Fn$ by summation using a `while` loop.**

# %%
# functions: -----------------------------------------------------------------
def fib_whl(n):
    if n == 0 or n == 1:
        return n
    else:
        fa = 0
        fb = 1
        fn = 1
        i = 2
        while i < n:
            fa = fb
            fb = fn
            fn= fa + fb
            i +=1
        return fn

    
    
# test result: --------------------------------------------------------------- 
test_func(fib_whl)


# %% [markdown]
# ***

# %% [markdown]
# **d. Write a function `fib_rnd()` with the same signature that computes $Fn$ using the rounding method described on the Wikipedia page linked above.**

# %%
# functions: -----------------------------------------------------------------
def fib_rnd(n):
    psi = (1+np.sqrt(5))/2
    return round(psi**n/np.sqrt(5))


# test result: ---------------------------------------------------------------  
test_func(fib_rnd)


# %% [markdown]
# ***

# %% [markdown]
# **e. Write a function `fib_flr()` with the same signature that computes $Fn$ using the truncation method described on the Wikipedia page linked above.**

# %%
# functions: -----------------------------------------------------------------
def fib_flr(n):
    psi = (1+np.sqrt(5))/2
    return np.floor(psi**n/np.sqrt(5)+0.5)


# test result: ---------------------------------------------------------------  
test_func(fib_flr)

# %% [markdown]
# ***

# %% [markdown]
# **f. For a sequence of increasingly large values of `n` compare the median computation time of each of the functions above. Present your results in a nicely formatted table. (Point estimates are sufficient).**

# %%
# functions list: ------------------------------------------------------------
func_name = ['fib_rec', 'fib_for', 'fib_whl', 'fib_rnd', 'fib_flr']
func_list = [fib_rec, fib_for, fib_whl, fib_rnd, fib_flr]
# 79: ------------------------------------------------------------------------

# %%
# Compute running time of each function with n = 30: -------------------------
count = 0
while count<5:
    for i in range(5):
        n = 30
        t1 = 1000*time.time()
        result = func_list[i](n)
        t2 =1000*time.time()
        print(func_name[i]+";n=" ,n , ";result=" ,result , ";time=",(t2 - t1))
    count += 1
# 79: ------------------------------------------------------------------------

# %%
# data frame: ----------------------------------------------------------------
dat = pd.DataFrame(
    {
     "function": func_name,
     "value of n": np.repeat(30,5),
     "median run time (ms)": [300.2,0.0,0.0,0.0,0.0]
     }
    )
# 79: ------------------------------------------------------------------------

# %% [markdown]
# *Comparison of computation time of functions above*

# %% [markdown]
# |    | function   |   value of n |   median run time (ms) |
# |---:|:-----------|-------------:|-----------------------:|
# |  0 | `fib_rec()`    |           30 |                  300.2 |
# |  1 | `fib_for()`    |           30 |                    0   |
# |  2 | `fib_whl()`    |           30 |                    0   |
# |  3 | `fib_rnd()`    |           30 |                    0   |
# |  4 | `fib_flr()`    |           30 |                    0   |

# %% [markdown]
# ***

# %% [markdown]
# ## Question 2 - Pascal’s Triangle [20]

# %% [markdown]
# a. Write a function to compute a specified row of Pascal’s triangle. An arbitrary row of Pascal’s triangle can be computed efficiently by starting with $\tbinom{n}{0} = 1$  and then applying the following recurrence relation among binomial coefficients, $$\tbinom{n}{k} = \tbinom{n}{k-1} * \frac{n+1-k}{k}$$

# %%
# function: ------------------------------------------------------------------
def row_PascT(n):
    C = 1 
    for i in range(1, n + 1):
        print(C, end = " ")
        C = int(C * (n - i) / i)
    print("")

    
# display the 5th row of Pascal Triangle:-------------------------------------
row_PascT(5)

# 79: ------------------------------------------------------------------------

# %% [markdown]
# ***

# %% [markdown]
# b. Write a function for printing the first $n$ rows of Pascal’s triangle using the conventional spacing with the numbers in each row staggered relative to adjacent rows. Use your function to display a minimum of 10 rows in your notebook.

# %%
# function: ------------------------------------------------------------------
def printPascT(n):    
    for line in range(1, n + 1):
        C = 1  ### the 1st element of each row is 1
        for i in range(1, line + 1):
            print(C, end = " ")
            C = int(C * (line- i) / i) ### applying the recurrence relation
        print("")

        
        
# print first 10 rows of Pascal Triangle: ------------------------------------

printPascT(10)

# 79: ------------------------------------------------------------------------

# %% [markdown]
# ***

# %% [markdown]
# ## Question 3 - Statistics 101 [40]

# %% [markdown]
# a. The standard point and interval estimate for the populaiton mean based on Normal theory takes the form $\bar{x}±z×se(x)$ where $\bar{x}$ is the mean, $se(x)$ is the standard error, and $z$ is a Gaussian multiplier that depends on the desired confidence level. Write a function to return a point and interval estimate for the mean based on these statistics.

# %% [markdown]
# **缺一个判断1d array 的imformative exception**

# %%
# function: ------------------------------------------------------------------
def get_estimate(data,conf_lv=0.95,input_string=True):
    """
    get point and interval estimate
    
    Parameter
    ---------
    data: 1d array or list
    conf_lv: confidence level, an integer range(0,1)
    
    Returns
    ---------
    A string of estimate or a dict
    """
    from scipy.stats import norm
    
    data = np.array(data)
    if type(data) is np.ndarray:
        theta_hat = np.mean(data)
        se = np.std(data)

        z = norm.ppf(0.5+conf_lv/2,loc=0,scale=1)
        CI_left = theta_hat- z * se
        CI_right = theta_hat+ z * se

        string = str(theta_hat)+" "+"["+str(conf_lv*100)+'% CI:'+'('+str(round(
                    CI_left,4))+','+str(round(CI_right,4))+')]'
        dic = {}
        dic['est'] = theta_hat
        dic['lwr'] = CI_left
        dic['upr'] = CI_right
        dic['level'] = conf_lv

        if input_string == True:
            return string
        if input_string == None:
            return dic
        else:
            return 'Parameter `input string` can only be True or None'
    else:
        return print('input data can only be 1d array or any object'+
                     'coercable to such an array using np.array()')
# 79: ------------------------------------------------------------------------


# %%
# Test: ----------------------------------------------------------------------
a = [1,2,3]
print(get_estimate(a,))
print(get_estimate(a,0.98))
print(get_estimate(a,0.95,input_string=None))
print(get_estimate(a,0.95,input_string=False))

# 79: ------------------------------------------------------------------------

# %% [markdown]
# b. There are a number of methods for computing a confidence interval for a population proportion arising from a Binomial experiment consisting of $n$ independent and identically distributed (iid) Bernoulli trials. Let $x$ be the number of successes in thes trials. In this question you will write a function to return point and interval estimates based on several of these methods. Your function should have a `parameter` method that controls the method used. Include functionality for each of the following methods.

# %%
# function: ------------------------------------------------------------------
def get_estimate_Bino(dat,conf_lv=0.95,method='Normal approximation'):
    
    """
    computing a point estimate confidence interval for a population proportion
    arising from a Binomial experiment consisting of n iid Bernoulli trials.
    
    Parameter
    ---------
    dat: a 1d numpy array that only have 0 and 1 
    conf_lv:confidence level
    method: 4 methods of computing CI {Normal approximation, CP, Jeffery, AC}
    
    Returns
    ---------
    A dict of point estimate and confidence interval
    """
    from scipy.stats import norm,beta
    import numpy as np
    
    dat = np.array(dat)
    if type(dat) is not np.ndarray:
        return 'input data must be 1d numpy array.'
    else:
        x = sum(dat)
        n = len(dat)
        p_hat = x/n
        dic = {}
        dic['est'] = p_hat
        dic['level'] = conf_lv

        if method =='Normal approximation':
            if min(x,n-x)>12:
                z = norm.ppf(0.5+conf_lv/2,loc=0,scale=1)
                dic['lwr'] = p_hat - z * np.sqrt(p_hat*(1-p_hat)/n)
                dic['upr'] = p_hat + z * np.sqrt(p_hat*(1-p_hat)/n)
                dic['method']='Normal approximation'
            else: 
                return 'condition for this method is not satisfied!'

        if method == 'CP':
            alpha = 1-conf_lv
            dic['lwr'] = beta.ppf(alpha/2, x, n-x+1)
            dic['upr'] = beta.ppf(1-alpha/2, x+1, n-x)
            dic['method']='Clopper-Pearson'

        if method == 'Jeffery':
            alpha = 1-conf_lv
            dic['lwr'] = max(0,beta.ppf(alpha/2, x+0.5, n-x+0.5))
            dic['upr'] = min(beta.ppf(1-alpha/2, x+0.5, n-x+0.5),1)
            dic['method']='Jeffery'

        if method == 'AC':
            z = norm.ppf(0.5+conf_lv/2,loc=0,scale=1)
            n_t = n + z**2
            p_t = (x+z**2/2)/n_t 
            dic['lwr'] = p_t - z * np.sqrt(p_t*(1-p_t)/n)
            dic['upr'] = p_t + z * np.sqrt(p_t*(1-p_t)/n)
            dic['method']='Agresti-Coull'

    return dic

# 79: ------------------------------------------------------------------------


# %% [markdown]
# c. Create a 1d Numpy array with $42$ ones and $48$ zeros. Construct a nicely formatted table comparing $90$, $95$, and $99%$ confidence intervals using each of the methods above (including part a) on this data. Choose the number of decimals to display carefully to emphasize differences. For each confidence level, which method produces the interval with the smallest width?

# %%
# Create a 1d Numpy array with 42 ones and 48 zeros---------------------------

a = np.repeat(0,42)
b = np.repeat(1,48)
dat =np.append(a,b)

# 79: ------------------------------------------------------------------------

# %%
# ALL 5 Methods result: ----------------------------------------------------------

# method in part a 
res_a_95 = get_estimate(dat,input_string=None)
res_a_90 = get_estimate(dat,conf_lv=0.9,input_string=None)
res_a_99 = get_estimate(dat,conf_lv = 0.99,input_string=None)

# 4 methods in part b
res_NA_90 = get_estimate_Bino(dat,conf_lv=0.9,method='Normal approximation')
res_NA_95 = get_estimate_Bino(dat,conf_lv=0.95,method='Normal approximation')
res_NA_99 = get_estimate_Bino(dat,conf_lv=0.99,method='Normal approximation')

res_CP_90 = get_estimate_Bino(dat,conf_lv=0.9,method='CP')
res_CP_95 = get_estimate_Bino(dat,conf_lv=0.95,method='CP')
res_CP_99 = get_estimate_Bino(dat,conf_lv=0.99,method='CP')

res_Jeffery_90 = get_estimate_Bino(dat,conf_lv=0.9,method='Jeffery')
res_Jeffery_95 = get_estimate_Bino(dat,conf_lv=0.95,method='Jeffery')
res_Jeffery_99 = get_estimate_Bino(dat,conf_lv=0.99,method='Jeffery')

res_AC_90 = get_estimate_Bino(dat,conf_lv=0.9,method='AC')
res_AC_95 = get_estimate_Bino(dat,conf_lv=0.95,method='AC')
res_AC_99 = get_estimate_Bino(dat,conf_lv=0.99,method='AC')
# 79: ------------------------------------------------------------------------

# %%
# Write result into a Table: ------------------------------------------------
methods_lst = ['Method in part a','Normal Approximation','Clopper-Pearson',
               'Jeffery','Agresti-Coull']*3

CI_lst = ['90%']*5+['95%']*5+['99%']*5

lwr_lst = [res_a_90['lwr'],res_NA_90['lwr'],res_CP_90['lwr'],res_Jeffery_90['lwr'],res_AC_90['lwr'],
           res_a_95['lwr'],res_NA_95['lwr'],res_CP_95['lwr'],res_Jeffery_95['lwr'],res_AC_95['lwr'],
           res_a_99['lwr'],res_NA_99['lwr'],res_CP_99['lwr'],res_Jeffery_99['lwr'],res_AC_99['lwr']]


upr_lst = [res_a_90['upr'],res_NA_90['upr'],res_CP_90['upr'],res_Jeffery_90['upr'],res_AC_90['upr'],
           res_a_95['upr'],res_NA_95['upr'],res_CP_95['upr'],res_Jeffery_95['upr'],res_AC_95['upr'],
           res_a_99['upr'],res_NA_99['upr'],res_CP_99['upr'],res_Jeffery_99['upr'],res_AC_99['upr']]

k =3 
res_df = pd.DataFrame(
    {
     "Methods": methods_lst,
     "confidence level": CI_lst,
     "lower bound": [round(i,k) for i in lwr_lst],
     "upper bound": [round(i,k) for i in upr_lst]
     }
    )
# 79: ------------------------------------------------------------------------

# %%
res_df['interval width'] = res_df['upper bound'] - res_df['lower bound']
res_df

# %% [markdown]
# **For all 3 confidence level, Jeffery's intervals has the smallest width.**

# %%
