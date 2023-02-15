# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 04:24:30 2023

@author: Marc
"""
import numpy as np

#want to add inflation here
def accumReturn(r,c,ic=0,i=0):  
    '''
    Gives the accumlated return for varying r,c, and i (ic is constant) and
    returns an array the same size at the parameters. i & c can be scalars 
    if they do not change. If r,c,i are all constant, then use
        calcReturnPmt(r,c,n,i) + calcReturnInit(r,ic,n,i)
    The return value is an array with the cumulative return for each period.
    If only the final value is of interest, then only use sum[-1]

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
        return np.array([0])
    
    #make sure i is an array
    if ((type(i)==int) or (type(i)==float)):
        iarr = i*np.ones(r.size)
    elif (type(i)==np.ndarray):
        iarr = i
    else:
        print('unknown data type for "c"')
        return np.array([0])
    
    sum = np.zeros(r.size)
    sum[0] = carr[0] + ic
    for k in range(1,r.size):
        sum[k] = carr[k] + sum[k-1]*(1+r[k])/(1+iarr[k])
    return sum

def calcReturnPmt(r,c,n,i=0):
    '''
    Gives the return after n years of a periodic deposits, c, given the 
    return rate r and inflation i. In its simplest form, this equation is 
    c/r*((1+r)^n - 1), one of the Time Value of Money equations. All the 
    parameters can be either scalars or arrays to allow r and i to vary over 
    the n periods. n can also be an array to give cumulative growth.
    
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
    Gives the return after n years of a single initial deposit, ic, given the 
    return rate r and inflation i. In its simplest form, this equation is 
    ic*(1+r)^n. r & i can be scalars or arrays to allow them to vary over the 
    n periods. All the parameters can be either scalars or arrays to allow
    r and i to vary over the n periods. n can also be an array to give 
    cumulative growth.
    
    
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

def taxBrackets(file='single'):
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

def standardDeducuction(file='single'):
    '''
    Parameters
    ----------
    file : str, optional
        filing status, one of 'single','joint','separate','head'.
        The default is 'single'.

    Returns
    -------
    standard deduction for 2022.
    '''
    if ('single'==file):
        ded = 12950
    elif ('joint'==file):
        ded = 25900
    elif ('separate'==file):
        ded = 12950
    else: #head of household
        ded = 19400
    return ded

def taxRate(inc,tb,ded=0):
    '''
    Calculate the effective tax rate given taxable income.
    This function uses a mathematically identical calculation of tax that is 
    algorithmically simpler. e.g. if income is $50,000 then
        tax = 50000*0.10 +
              (50000-10275)*(0.12-0.1) +
              (50000-41775)*(0.22-0.12)
    The more common calculation is 
        tax = 10275*0.10 +
              (41775-10275)*0.12 +
              (50000-41775)*0.22
    Parameters
    ----------
    inc : float or int
        taxable income in $.
    tr : array of tax brackets and tax rates
    ded : float or int
        deduction, taxable income is gross income - deduction 
        If effective tax = tax/total income, then use taxRate(inc,tb,ded), but
        if effective tax = tax/taxable income, then set ded to zero
    Returns
    -------
    taxRate : float
        The effective tax rate.
    '''        
    txInc = inc-ded
    k=1; tax = txInc*tb[0,1]
    while ((k<tb.shape[0]) and (txInc>tb[k,0])):
        tax = tax+(txInc-tb[k,0])*(tb[k,1]-tb[k-1,1])
        k=k+1
    return tax/inc

def taxRate2(inc,tb,ded=0):
    '''
    Calculate the effective tax rate given taxable income.
    This function uses the algorithm below (assuming income is $50k) 
        tax = 10275*0.10 +
              (41775-10275)*0.12 +
              (50000-41775)*0.22
    Parameters
    ----------
    inc : float or int
        taxable income in $.
    tr : array of tax brackets and tax rates
    ded : float or int
        deduction, taxable income is gross income - deduction 
        If effective tax = tax/total income, then use taxRate(inc,tb,ded), but
        if effective tax = tax/taxable income, then set ded to zero
    Returns
    -------
    taxRate : float
        The effective tax rate.
    '''        
    txInc = inc-ded
    if (txInc<=tb[1,0]):
        tax=txInc*tb[0,1]
    else:
        k=1; tax = tb[1,0]*tb[0,1]
        while ((k<tb.shape[0]-1) and (txInc>tb[k,0])):
            if (txInc>tb[k+1,0]):
                tax = tax+(tb[k+1,0]-tb[k,0])*tb[k,1]
            else:
                tax = tax+(txInc-tb[k,0])*tb[k,1]
            k=k+1
        if (txInc>tb[tb.shape[0]-1,0]):
            tax = tax+(txInc-tb[k,0])*tb[k,1]
    return tax/txInc

def taxedRetirement(r,c,n,i,Tw,Tr):  
    '''
    Calculate the after-tax money available in retirement for an account with 
    periodic contributions assuming a constant rate of return, 
    constant contributions, and constant inflation. This is designed to 
    compare the effective savings of a normal investment account versus
    Roth and traditional tax sheltered accounts.
    For a normal account, use Tw=working years ordinary tax, Tr=capital gains tax
    For traditional IRA/401k, use Tw = 0, Tr=retirement ordinary tax
    For Roth IRA/401k, use Tw=working years ordinary tax, Tr=0
    
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
