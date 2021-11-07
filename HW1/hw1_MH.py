# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   markdown:
#     extensions: footnotes
# ---

# ## Homework 1
# **Mengtong Hu **  
# *September 9, 2021*
# 
import math
import time
import statistics
import pandas as pd
from IPython.core.display import display, HTML
from scipy import stats 
import numpy as np


# ## Question 1

# ***
# This is _question 0_ for [problem set 1](https://jbhender.github.io/Stats507/F21/ps/ps1.html) of [Stats 507](https://jbhender.github.io/Stats507/F21/).
# <blockquote>Question 0 is about Markdown.</blockquote>

# The next question is about the __Fibonnaci sequence__, $F_n=F_{n−2}+F_{n−1}$. In part __a__ we will define a Python function `fib_rec()`.
# <br>
# Below is a ...
# ### Level 3 Header
# Next, we can make a bulleted list:
# - Item 1
#     * detail 1
#     * detail 2
# - Item 2
# <br>
# Finally, we can make an enumerated list:
# 1. Item 1
# 2. Item 2
# 3. Item 3
# ***

# ## Question 2 - Fibonnaci Sequence

# We are going to present five ways to calculate Fibonnaci Sequence and test their correctness using `test_fib()`

def test_fib(funcs):
    """
    Use three test cases to test the correctness of the Fibonnaci Sequence functions:
    
    Parameters
    ----------
    funcs : Different Fibonnaci functions to test


    Returns
    -------
    If the function passes all three test cases, pring the message

    """
    assert funcs(7) == 13
    assert funcs(11) == 89
    assert funcs(13) == 233
    return (funcs.__name__ + " has passed all the test caes " )


# - A recursive function calcuating Fibonnaci sequence

def fib_rec(n):
    """
    A recursive function calcuating Fibonnaci sequence

    Parameters
    ----------
    n : integer
        the nth Fibonnaci number in the sequence

    Returns
    -------
    recursively calling fib_rec to calculte Fibonnaci number.
    Will return the nth Fibonnaci number

    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    return(fib_rec(n-2) + fib_rec(n-1))

# - Test recursive function `fib_rec()`

print(test_fib(fib_rec))


# - computes $F_n$ by summation using a for loop.

def fib_for(n):
    """
    Compute Fibonnaci sequence using a for loop

    Parameters
    ----------
    n : integer
        the nth Fibonnaci number in the sequence

    Returns
    -------
    the nth Fibonnaci number in the sequence

    """
    res = [0, 1]
    for i in range(n-1):
        res.append(res[i] + res[i+1])
    return res[n]

# - Test `fib_for()`

print(test_fib(fib_for))


# - computes $F_n$ by summation using a while loop.

def fib_whl(n):
    """
    Compute Fibonnaci sequence using a while loop

    Parameters
    ----------
    n : integer
        the nth Fibonnaci number in the sequence

    Returns
    -------
    the nth Fibonnaci number in the sequence

    """
    res = [0, 1]
    i = 0
    while i<n:
        res.append(res[i] + res[i+1])
        i = i + 1
    return res[n]


# - Test recursive function `fib_for()`

print(test_fib(fib_whl))

# - computes $F_n$ using the rounding method 
# <br>
# We need to first define the constant $\phi$

phi = (1 + math.sqrt(5)) / 2

def fib_rnd(n):
    """
    Compute Fibonnaci sequence using the rounding function

    Parameters
    ----------
    n : integer
        the nth Fibonnaci number in the sequence

    Returns
    -------
    the nth Fibonnaci number in the sequence

    """
    return (int(round(phi ** n / math.sqrt(5), ndigits=(0))))

# - Test recursive function `fib_rnd()`

print(test_fib(fib_rnd))  


def fib_flr(n):
    """
    Compute Fibonnaci sequence using the flooring function

    Parameters
    ----------
    n : integer
        the nth Fibonnaci number in the sequence

    Returns
    -------
    the nth Fibonnaci number in the sequence

    """
    return (int(phi ** n / math.sqrt(5) + 1/2))    

# - Test flooring function `fib_flr()`

print(test_fib(fib_flr))   

# Now, we benchmark the running time of the three methods. For each function `fib_***()`and __n__ , we repeat the procedure 10 times and take the median runtime of the 10 repetitions.

def time_fib(funcs, n):
    """
    Compute the median computation time of a Fibonnaci function

    Parameters
    ----------
    n : integer
        the nth Fibonnaci number in the sequence
    funcs : Different Fibonnaci functions to test

    Returns
    -------
    the median runtime in ms over 10 repetitions 
    of the same function call

    """
    time_res = []
    
    for i in range(10):
        start = time.perf_counter()
        funcs(n) 
        time_res.append(time.perf_counter() - start)
    return ( 1000 * statistics.median(time_res) )

# Use a dictionay `process_time` to store the median time of each of the five methods:

process_time = {'Recursive':[],
                'For Loop':[],
                'While Loop':[],
                'Rounding':[],
                'Truncating':[]}   
for n in [5*i for i in range(1, 6)]:
    process_time['Recursive'].append(time_fib(fib_rec,n))
    process_time['For Loop'].append(time_fib(fib_for,n))
    process_time['While Loop'].append(time_fib(fib_whl,n))
    process_time['Rounding'].append(time_fib(fib_rnd,n))
    process_time['Truncating'].append(time_fib(fib_flr,n))


# Display the resutls in a table

process_time_tb = pd.DataFrame.from_dict(process_time, orient = 'index',
                                         columns=['n = 5', 'n = 10', 'n = 15', 'n =20', 'n = 25'])
display(HTML(process_time_tb.to_html()))


# The table presents the median run time in milliseconds for five methods we considered for $n = 5, 10 \dots 25$. 
# - Recusive funtion is the slowest and its run time grows really fast, depending on the sample size
# - Trucating method is the fastest and does not depend on the sample size. Rounding method has similar perforances.
# - For loop and While loop process slower as n increases

# ## Question 3 - Pascal’s Triangle

# In `compute_pascal`, we compute an arbitrary row of the Pascal’s Triangle

def compute_pascal(n):
    """
    Compute the nth row of Pascal’s triangle

    Parameters
    ----------
    n : integer
        which row too compute

    Returns
    -------
    pascal_n : a list of integers
        The nth row of Pascal’s triangle.

    """
    pascal_n = [1]
    prev = 1
    for k in range(1,n+1):
        cur = prev * (n+1-k)/k
        pascal_n.append(int(cur))
        prev = cur
    return(pascal_n)
print(compute_pascal(3))


# In `print_pascal`, we print the first __n__ rows of the Pascal’s Triangle.

def print_pascal(n):
    """
    print the firt n rows of Pascal’s triangle with proper spacing

    Parameters
    ----------
    n : integer
        The total numbers of rows 

    Returns
    -------
    None, but will print output
        First n rows of Pascal’s triangle.

    """
    for i in range(n) :
        total_width = n * 4 + 1
        front_space = (total_width - (i * 4)) // 2
        print(' ' * front_space, end = '')
        print('   '.join(map(str,compute_pascal(i))))

    return

# Using `print_pascal` to print first 10 rows of the pascal triangle

print_pascal(10)

# ### Question 3
# #### part a

def ci_mean(a, level, output_format= True):
    """
    print the firt n rows of Pascal’s triangle with proper spacing

    Parameters
    ----------
    a : any obejct
        the data which we constrcut confidence interval for
    output_format : string, optional
             determine the output format
             Set to 'None' if want to return a dicionary. The Default is True.
                 
    level : int
            confidence interval


    Returns
    -------
    res : string, Default
          preformatted
    res : dictionary, when format = None or 'None'
          with keys 'est', 'lwr', 'upr', and  'level'
          
    """
    try:
        a = np.array(a)
        a.astype(float)

    except ValueError:         
        print ('ValueError: Data input invalid. Use 1d Numpy arrary or',
               'objects are coercable to 1d Numpy arrary.')
        return

    n = len(a)
    # ddof parameter set to 1 in np.std() gives the samples standard deviation
    se = np.std(a, ddof=1) / math.sqrt(n)
    tail_prop = (100 - level) / 2 / 100
    z_val = - stats.norm.ppf(tail_prop)
    mean = sum(a) / n
    lwr, upr  = mean - z_val * se, mean + z_val * se
    res = {'est' : mean, 'lwr' : lwr, 
          'upr' : upr, 'level' : level}
    ci_str = "{est:.3f}[{level:.0f}%CI :  ({lwr:.3f},{upr:.3f})]"
    #unpack the values of the dictionary and apply the format
    formatted_res = ci_str.format(**res)


    if output_format == True :
        res = formatted_res
    
    return(res)

# When the input format is an np.arrary:

print(ci_mean(np.array([12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29]), 95))
print(ci_mean(np.array([12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29]), 95, output_format = None))

# When the input format is not an array of float or int:

ci_mean("haa", 95)


# #### part b

def ci_proportion(a, level , Method, output_format = True ):
    """
    print the firt n rows of Pascal’s triangle with proper spacing

    Parameters
    ----------
    a : any obejct
        the data which we constrcut confidence interval for
    output_format : string, optional
             determine the output format
             Set to 'None' if want to return a dicionary. The Default is True.
                 
    level : int
            confidence interval
    Method : string
             the method used to compute the CI. Possible values include 
             'Standard', 'Clopper-Pearson', 'Jeffrey', 'Agresti-Coull'

    Returns
    -------
    res : string, Default
          preformatted
    res : dictionary, when format = 'None'
          with keys 'est', 'lwr', 'upr', and  'level'
          
    """
    try:
        a = np.array(a)
        a.astype(float)
    except ValueError:         
        print('TypeError: Data input invalid. Use 1d Numpy arrary or objects are coercable to 1d Numpy arrary')
        return()
    #n :   the number of independent and iid Bernoulli trials
    # x :  the number of sucesses
    n = len(a)
    x = sum(a)
    p = x / n
    alpha = 1 - level / 100
    tail_prop = (100 - level) / 2 / 100
    z_val = - stats.norm.ppf(tail_prop)
    if Method == "Standard":
        if n * p <= 12 or n * (1 - p) <= 12 :
            raise ValueError('Proportion or same size too small and',
                             'not satisfying n * p > 12 and n * (1 - p) > 12')
        else: 
            se = math.sqrt(p * (1-p) / n)
            lwr, upr  = p - z_val * se, p + z_val * se
                             
    if Method == "Clopper-Pearson":       
        lwr, upr = stats.beta.ppf(alpha /2 , x , n - x + 1) , \
                   stats.beta.ppf(1 - alpha /2 , x + 1 , n - x)
                             
    if Method == "Jeffrey":
        lwr, upr = max(stats.beta.ppf(alpha /2 , x , n - x + 1), 0) , \
                   min(stats.beta.ppf(1 - alpha /2 , x + 1 , n - x), 1)
        
    if Method == "Agresti-Coull":
        n = n + z_val ** 2
        x = x + z_val ** 2 / 2
        p = x / n
        se = math.sqrt(p * (1-p) / n)
        lwr, upr  = p - z_val * se, p + z_val * se
   
    
   
    
    res = {'est' : p, 'lwr' : lwr, 
          'upr' : upr, 'level' : level}
    ci_str = "{est:.3f}[{level:.0f}%CI :  ({lwr:.3f},{upr:.3f})]"
    formatted_res = ci_str.format(**res)


    if output_format == True :
        res = formatted_res
    
    return(res)



# - Create a 1d Numpy array with 42 ones and 48 zeros. 

input_one_zero = np.ones(42)
input_one_zero = np.pad(input_one_zero, (0, 48), 'constant')
print(input_one_zero)

# - Construct a nicely formatted table comparing $90\%$, $95\%$, and $99\%$ confidence intervals using each of the methods above (including part a) on this data.
# - For each confidence level, which method produces the interval with the smallest width?

compare_ci = {'Normal theory for mean':[],
                'Normal approximation for binomial distribution':[],
                'Clopper-Pearson':[],
                'Jeffrey':[],
                'Agresti-Coull':[]}   
for level in [90, 95, 99]:
    ci_proportion(input_one_zero, level, Method = 'Standard')
    compare_ci['Normal theory for mean'].append(ci_mean(input_one_zero, level))
    compare_ci['Normal approximation for binomial distribution'].append(ci_proportion(input_one_zero, level, Method = 'Standard'))
    for method in ['Clopper-Pearson','Jeffrey','Agresti-Coull']:
        compare_ci[method].append(ci_proportion(input_one_zero, level, Method = method))


compare_ci_tb = pd.DataFrame.from_dict(compare_ci, orient = 'index',
                                         columns=['confidence level = 90', 'confidence level = 95', 'confidence level = 99'])
display(HTML(compare_ci_tb.to_html()))


# - For all Confidence levels, Agresti-Coull gives the smallest width
#
