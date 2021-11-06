# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
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
def test_fib(funcs):
    assert funcs(7) == 13
    assert funcs(11) == 89
    assert funcs(13) == 233
    return ("All the test caes have paseed for " + funcs.__name__)
    
def fib_rec(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return(fib_rec(n-2) + fib_rec(n-1))

print(test_fib(fib_rec))


def fib_for(n):
    res = [0, 1]
    for i in range(n-1):
        res.append(res[i] + res[i+1])
    return res[n]
print(test_fib(fib_for))

def fib_whl(n):
    res = [0, 1]
    i = 0
    while i<n:
        res.append(res[i] + res[i+1])
        i = i + 1
    return res[n]
print(test_fib(fib_whl))


phi = (1 + math.sqrt(5)) / 2

def fib_flr(n):
    return (int(phi ** n / math.sqrt(5) + 1/2))    

print(test_fib(fib_flr))   

def fib_rnd(n):
    return (int(round(phi ** n / math.sqrt(5), ndigits=(0))))
print(test_fib(fib_rnd))  

#Now we bench mark the three method.
def time_fib(funcs, n):
    time_res = []
    
    for i in range(10):
        start = time.perf_counter()
        funcs(n) 
        time_res.append(time.perf_counter() - start)
    return ( 1000 * statistics.median(time_res) )
   
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


print(process_time)
process_time_tb = pd.DataFrame.from_dict(process_time, orient = 'index',
                                         columns=['n = 5', 'n = 10', 'n = 15', 'n =20', 'n = 25'])
display(HTML(process_time_tb.to_html()))
### 

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

print_pascal(10)

### Question 3
## part a
def check_input(a) :
    """
    Check if the input is 1-d numpy array or coercable to 1-d array

    Parameters
    ----------
    a : any obejct

    Returns
    -------
    res : a 1-d numpy array
          or raise an error
          
    """
    
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
    res : dictionary, when format = 'None'
          with keys 'est', 'lwr', 'upr', and  'level'
          
    """
    try:
        a = np.array(a)
        a.astype(float)

    except ValueError:         
        print ('ValueError: Data input invalid. Use 1d Numpy arrary or objects are coercable to 1d Numpy arrary.')
        return

    n = len(a)
    # ddof parameter set to 1 in np.std() gives the samples standard deviation
    se = np.std(a, ddof=1) / math.sqrt(n)
    tail_prop = (100 - level) / 2 / 100
    z_val = - stats.norm.ppf(tail_prop)
    mean = sum(a) / n
    lwr, upr  = mean - z_val * se, mean + z_val * se
    res = {'est' : round(mean, ndigits = 2), 'lwr' : round(lwr, ndigits = 2), 'upr' : round(upr, ndigits = 2), 'level' : level}
    ci_str = "{est:.2f}[{level:.0f}%CI :  ({lwr:.2f},{upr:.2f})]"
    #unpack the values of the dictionary and apply the format
    formatted_res = ci_str.format(**res)
    if output_format :
        res = formatted_res
    
    return(res)

print(ci_mean(np.array([12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29]), 95))
print(ci_mean("haa", 95))
np.asarray("haa")
## part b
def ci_proportion(a, level , Method,output_format = True ):
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
             the method used to compute the CI. Possible values include 'Standard', 'Clopper-Pearson', 'Jeffrey', 'Agresti-Coull'

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
    n = len(a)
    x = sum(a)
    p = x / n
    alpha = 1 - level / 100
    if Method == "Standard":
        if n * p <= 12 or n * (1 - p) <= 12 :
            raise ValueError("Proportion or same size too small and not satisfying n * p > 12 and n * (1 - p) > 12")
        else: 
            tail_prop = (100 - level) / 2 / 100
            z_val = - stats.norm.ppf(tail_prop)
            se = math.sqrt(p * (1-p) / n)
            lwr, upr  = p - z_val * se, p + z_val * se
    if Method == "Clopper-Pearson":
        
        lwr, upr = stats.beta.ppf(alpha /2 , x , n - x + 1) , stats.beta.ppf(1 - alpha /2 , x + 1 , n - x)
    if Method == "Jeffrey":
        lwr, upr = max(stats.beta.ppf(alpha /2 , x , n - x + 1), 0) , min(stats.beta.ppf(1 - alpha /2 , x + 1 , n - x), 1)
        
   
    
   
    
   
    
   res = {'est' : round(mean, ndigits = 2), 'lwr' : round(lwr, ndigits = 2), 'upr' : round(upr, ndigits = 2), 'level' : level}
    ci_str = "{est:.2f}[{level:.0f}%CI :  ({lwr:.2f},{upr:.2f})]"
   #n :   the number of independent and iid Bernoulli trials
   # x :  the number of sucesses
   