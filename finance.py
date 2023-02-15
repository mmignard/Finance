# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 04:24:30 2023

@author: Marc
"""
import numpy as np

#want to add inflation here
def accumReturn(r,c,ic=0,i=0):  
    '''
    Parameters
    ----------
    r : numpy.ndarray
        Fractional return for each time period.
    c : float or numpy.ndarray
        Contribution for each time period.
    ic : float
        Intitial contibution.
    i : float or numpy.ndarray
        inflation rate.
    Note: r,c,i must have same number of elements.
    Returns
    -------
    sum : numpy.ndarray
        Calculate the accumulated return on an investment given arrays of
        returns, periodic contribution, inflation, and an initial contribution.
    '''  
    #make sure c is an array
    if ((type(c)==int) or (type(c)==float)):
        carr = c*np.ones(r.size)
    elif (type(c)==np.ndarray):
        carr = c
    else:
        print('unknown data type for "c"')
        return 0
    
    #make sure i is an array
    if ((type(i)==int) or (type(i)==float)):
        iarr = i*np.ones(r.size)
    elif (type(i)==np.ndarray):
        iarr = i
    else:
        print('unknown data type for "c"')
        return 0
    
    sum = np.zeros(r.size)
    sum[0] = carr[0] + ic
    for k in range(1,r.size):
        sum[k] = carr[k] + sum[k-1]*(1+r[k])/(1+iarr[k])
    return sum

def calcReturnPmt(r,c,n,i=0):
    '''
    Parameters
    ----------
    r : float or numpy.ndarray
        return rate for each time period if array, or 
        return rate for all time periods if float.
    c : float or numpy.ndarray
        Contribution for each time period if array, or 
        contribution for all time periods if float.
    n : float or numpy.ndarray
        Number of time periods to calculate cumulative value.
        If is array, can use to create a cumulative graph.
    i : float or numpy.ndarray
        inflation rate.
    Note: r,c,n,i must be float or have same number of elements.
    Returns
    -------
    sum : float or numpy.ndarray
        Cumulative return after n periods using Time Value of Money (TVM) equation.
    '''
    sum = c*(1+i)/(r-i)*(np.power((1+r)/(1+i),n+1) - 1)
    return sum

def calcReturnInit(r,ic,n,i=0):   
    '''
    Parameters
    ----------
    r : float or numpy.ndarray
        return rate for each time period if array, or 
        return rate for all time periods if float.
    ic : float
        Initial contribution.
    n : float or numpy.ndarray
        Number of time periods to calculate cumulative value.
        If is array, can use to create a cumulative graph.
    i : float or numpy.ndarray
        inflation rate.
    Note: r,i must be float or have same number of elements.
    Returns
    -------
    sum : float or numpy.ndarray
        cumulative return after n periods using constant return equation.
    '''
    sum = ic*np.power((1+r)/(1+i),n)
    return sum

def taxBrackets(file):
    '''
    Parameters
    ----------
    file : str, optional
        filing status, one of 'single','joint','separate','head'.
        The default is 'single'.

    Returns
    -------
    array of tax brackets and tax rates for 2022.
    '''
    if ('single'==file):
        tr = np.array([
            [0,0.1],
            [10275,0.12],
            [41775,0.22],
            [89075,0.24],
            [170050,0.32],
            [215950,0.35],
            [539900,0.37]])
    elif ('joint'==file):
        tr = np.array([
            [0,0.1],
            [20550,0.12],
            [83550,0.22],
            [178150,0.24],
            [340100,0.32],
            [431900,0.35],
            [647850,0.37]])
    elif ('separate'==file):
        tr = np.array([
            [0,0.1],
            [10275,0.12],
            [41775,0.22],
            [89075,0.24],
            [170050,0.32],
            [215950,0.35],
            [323925,0.37]])
    else: #head of household
        tr = np.array([
            [0,0.1],
            [14650,0.12],
            [55900,0.22],
            [89050,0.24],
            [17050,0.32],
            [215950,0.35],
            [539900,0.37]])
    return tr

def taxRate(inc,tb):
    '''
    Calculate the effective tax rate given taxable income
    Parameters
    ----------
    inc : float or int
        taxable income in $.
    tr : array of tax brackets and tax rates

    Returns
    -------
    taxRate : float
        The effective tax rate.
    '''        
    k=1; tax = inc*tb[0,1]
    while ((k<tb.shape[0]) and (inc>tb[k,0])):
        tax = tax+(inc-tb[k,0])*(tb[k,1]-tb[k-1,1])
        k=k+1
    return tax/inc

#An alternate way of calculating effective tax rate. This code is
#longer, and it doesn't seem to be any clearer
# def taxRate2(inc,tr):
#     if (inc<=tr[1,0]):
#         tax=inc*tr[0,1]
#     else:
#         k=1; tax = tr[1,0]*tr[0,1]
#         while ((k<tr.shape[0]-1) and (inc>tr[k,0])):
#             if (inc>tr[k+1,0]):
#                 tax = tax+(tr[k+1,0]-tr[k,0])*tr[k,1]
#             else:
#                 tax = tax+(inc-tr[k,0])*tr[k,1]
#             k=k+1
#         if (inc>tr[tr.shape[0]-1,0]):
#             tax = tax+(inc-tr[k,0])*tr[k,1]
#     return tax/inc

def taxedRetirement(r,c,n,i,Tw,Tr):  
    '''
    Parameters
    ----------
    r : float
        return rate from investments (per year, or 
        annual return/number of periods per year).
    c : float
        periodic contribution per period (annual?).
    n : float
        number of periods (years?).
    i : float
        inflation rate over period (per year, or 
        annual inflation/number of periods per year).
    Tw : float
        Tax rate while working.
    Tr : float
        Tax rate after retirement.
    Returns
    -------
    Sn : float
        Savings after all taxes are paid.
    '''
    Sn = (1-Tr)*(1-Tw)*calcReturnPmt(r,c,n,i) + Tr*Tw*n*c
    return Sn
