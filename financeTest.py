# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 09:18:04 2023

@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt

import finance as fn

saveImages = False
annotateSize = 8
fpath = './/images//'

##########################################################################
###    Comparison between initial lump contribution and annual contribution
###    
##########################################################################

cPmt = 5
r = 0.05
year = np.linspace(0,50,51)
n = 10
cL = 50

plt.figure(figsize=(5,3.5),dpi=300)
plt.subplot()
plt.title('Annual Contribution Versus Initial Lump Sum')
plt.plot(year,fn.calcReturnInit(r,cL,year),'k',label='initial contribution')
plt.plot(year,fn.calcReturnPmt(r,cPmt,year),'k--',label='annual contribution')
#plt.plot(year,cPmt*year,'k:',label='cumulative\nannual')
plt.ylabel('total savings\n($/1000)')
plt.xlabel('years')
#plt.yscale('log')
plt.xlim([year[0],year[-1]])
plt.ylim([0,1000])
plt.legend(loc=(0.01, 0.3))
#plt.text(22,1000,'PMT=${}k, n={},r={}\ninital lump\ncontribution=${:.1f}k'.format(cPmt,n,r,cL), fontsize=annotateSize)
plt.text(2,660,'inital lump contribution=${:.1f}k\nwith annual contribution=${}k,\ntotal contribution after {} years=${:.0f}k,\nCAGR={}%'.format(cL,cPmt,n,n*cPmt,100*r), fontsize=annotateSize)

plt.grid(True)
if saveImages:
    plt.savefig(fpath+"AnnualVsInitialContribution.svg", bbox_inches='tight')

plt.show()


##########################################################################
###     Continuous compounding
###     
##########################################################################

i = 0.2
n=1001
years = 10
contT = np.linspace(0,years,n)
contInt = np.exp(i*contT)

yearT = np.floor(contT,dtype=float)
y = np.power(1+i,yearT)

quartT = np.floor(contT*4,dtype=float)/4
quartInt = np.power(1+i/4,4)-1
q = np.power(1+quartInt,quartT)

monthT = np.floor(contT*12,dtype=float)/12
monthInt = np.power(1+i/12,12)-1
m = np.power(1+monthInt,monthT)

plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Effects of interest compounding period')
#plt.plot(contT,contInt,'k--',label='continuous')
plt.plot(contT,contInt,'k',linewidth=2,label='continuous')
plt.plot(contT,q,'k',linewidth=1,label='quarterly')
plt.plot(contT,y,'k--',label='yearly')
#plt.plot(contT,m,'k',label='monthly')
plt.grid(True)
plt.legend()
plt.ylim([0,8])
plt.xlim([0,10.02])
#plt.yscale('log')
plt.xlabel('year')
plt.ylabel('return/$')
if (saveImages):
    plt.savefig(fpath+"CompoundingInterval.svg", bbox_inches='tight')
plt.show()

##########################################################################
###    length of time retired before money runs out
###    
##########################################################################
# def retFrac(r,n):
#     return r/(1+r)*np.power(1+r,n)/(np.power(1+r,n)-1)
def retFrac(r,n,i=0.03):
    return (r-i)/(1+r)*np.power((1+r)/(1+i),n)/(np.power((1+r)/(1+i),n)-1)
n=np.linspace(10,40,31)

plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Maximum income from an investment account\nwith no return')
plt.plot(n,100/n,'k--',linewidth=1)
# r=0.0; i=0.03 #two ways of calculating effect of inflation (same results)
# plt.plot(n,100*retFrac(r,n,i),'k:o',linewidth=1,markersize=3,label='r={:.0f}%, i={:.0f}%'.format(r*100,i*100))
# plt.plot(n,100*i/(np.power(1+i,n)-1),'k--',linewidth=1,label='r={:.0f}%, i={:.0f}%'.format(r,i))
plt.grid(True)
plt.ylim([100*0.02,100*0.16])
plt.xlim([n[0],n[-1]])
plt.xlabel('years in retirement')
plt.ylabel('annual income as percent of\ninitial savings (w/vR0)')
y=20; f = 100/y
plt.plot([y,y],[2,f],'k:',linewidth=1)
plt.plot([10,y],[f,f],'k:',linewidth=1)
plt.text(10.5,f+0.1,'{:0.1f}%'.format(f), fontsize=annotateSize)
y=30; f = 100/y
#plt.annotate('', xy=(30,2), xytext=(30,f), arrowprops=dict(facecolor='black', linestyle=':', width=0.5, headwidth=0, headlength=1), fontsize=annotateSize)
#plt.annotate('', xy=(10,f), xytext=(30,f), arrowprops=dict(facecolor='black', linestyle=':', width=0.5, headwidth=0, headlength=1), fontsize=annotateSize)
plt.plot([y,y],[2,f],'k:',linewidth=1)
plt.plot([10,y],[f,f],'k:',linewidth=1)
plt.text(10.5,f+0.1,'{:0.1f}%'.format(f), fontsize=annotateSize)
if (saveImages):
    plt.savefig(fpath+"MaxIncomeNoReturn.svg", bbox_inches='tight')
plt.show()

#show increase due to return and decrease due to inflation
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Maximum income from an investment account\n with return and inflation')
r=0.10; i=0
plt.plot(n,100*retFrac(r,n,i),'k-o',linewidth=1,markersize=3,label='r={:.0f}%, i={:.0f}%'.format(r*100,i*100))
r=0.10; i=0.03
plt.plot(n,100*retFrac(r,n,i),'k-s',linewidth=1,markersize=3,label='r={:.0f}%, i={:.0f}%'.format(r*100,i*100))
plt.plot(n,100/n,'k--',linewidth=1,label='r={:.0f}%, i={:.0f}%'.format(0,0))
plt.grid(True)
plt.legend()
plt.ylim([100*0.02,100*0.16])
plt.xlim([n[0],n[-1]])
r=0.05; i=0.03
plt.xlabel('years in retirement')
plt.ylabel('annual income as percent of\ninitial savings (w/vR0)')
y=26.5
plt.annotate('', xy=(y,100*retFrac(0.1,y,0.03)), xytext=(y,100*retFrac(0.1,y,0)), arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=8), fontsize=annotateSize)
plt.text(y+0.5,8.1,'reduction due\nto inflation', fontsize=annotateSize)
y=21.5
plt.annotate('', xy=(y,100*retFrac(0.1,y,0)), xytext=(y,100/y), arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=8), fontsize=annotateSize)
plt.text(19.5,11,'increase due\nto return rate', fontsize=annotateSize)
if (saveImages):
    plt.savefig(fpath+"MaxIncomeWithReturn1.svg", bbox_inches='tight')
plt.show()

#show reasonable amounts to expect
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Maximum income from an investment account\n with return and inflation')
r=0.05; i=0
plt.plot(n,100*retFrac(r,n,i),'k-o',linewidth=1,markersize=3,label='r={:.0f}%, i={:.0f}%'.format(r*100,i*100))
r=0.05; i=0.03
plt.plot(n,100*retFrac(r,n,i),'k-s',linewidth=1,markersize=3,label='r={:.0f}%, i={:.0f}%'.format(r*100,i*100))
plt.plot(n,100/n,'k--',linewidth=1,label='r={:.0f}%, i={:.0f}%'.format(0,0))
plt.grid(True)
plt.legend()
plt.ylim([100*0.02,100*0.16])
plt.xlim([n[0],n[-1]])
r=0.05; i=0.03
plt.xlabel('years in retirement')
plt.ylabel('annual income as percent of\ninitial savings (w/vR0)')
y=20; f = 100*retFrac(r,y,i) #the last one with r=5%, i=3%
plt.plot([y,y],[2,f],'k:',linewidth=1)
plt.plot([10,y],[f,f],'k:',linewidth=1)
plt.text(10.5,f+0.1,'{:0.1f}%'.format(f), fontsize=annotateSize)
y=30; f = 100*retFrac(r,y,i) #the last one with r=5%, i=3%
plt.plot([y,y],[2,f],'k:',linewidth=1)
plt.plot([10,y],[f,f],'k:',linewidth=1)
plt.text(10.5,f+0.1,'{:0.1f}%'.format(f), fontsize=annotateSize)
if (saveImages):
    plt.savefig(fpath+"MaxIncomeWithReturn2.svg", bbox_inches='tight')
plt.show()
r=0.05; i=0.03
100*retFrac(r,n,i)

#estimate age
#First get the data in a reasonable format. Once this is done, don't need to do it again.
# fn = './/data//SurvivalFunctionRaw.txt'
# sf = np.reshape(np.genfromtxt(fn,delimiter='\t',dtype=float),(120,6))
# np.savetxt('.//data//SurvivalFunctionData.txt',sf,fmt='%.3f',delimiter=',')
sf = np.genfromtxt('.//data//SurvivalFunctionData.txt',delimiter=',',skip_header=3,unpack=True,dtype=float)

plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Life Survival Function, linear')
plt.plot(sf[0,:],sf[5,:],'k--',label='year 2100')
plt.plot(sf[0,:],sf[4,:],'k-',label='year 2050')
plt.plot(sf[0,:],sf[3,:],'k:',label='year 2000')
plt.plot(sf[0,:],sf[1,:],'k-.',label='year 1900')
plt.grid(True,which='both',)
plt.legend()
plt.ylim([-1,101])
plt.xlim([0,120])
plt.xlabel('age')
plt.ylabel('% probability of living to age')
if (saveImages):
    plt.savefig(fpath+"SurvivalFuncLin.svg", bbox_inches='tight')
plt.show()

import matplotlib as mpl
#remove trailing zeros in ylabel
def rmZeros(x,pos):
#    myStr = '{:0.3f}'.format(x)
    myStr = f'{x:0.3f}'
    while (myStr[-1] == '0'):
        myStr = myStr[:-1]
    return myStr

plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Life Survival Function, logarithmic')
plt.plot(sf[0,:],sf[5,:],'k--',label='year 2100')
plt.plot(sf[0,:],sf[4,:],'k-',label='year 2050')
plt.plot(sf[0,:],sf[3,:],'k:',label='year 2000')
plt.plot(sf[0,:],sf[1,:],'k-.',label='year 1900')
plt.grid(True,which='both')
plt.legend()
plt.yscale('log')
plt.ylim([1e-3,105])
plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(rmZeros))
plt.xlim([0,120])
plt.xlabel('age')
plt.ylabel('% probability of living to age')
if (saveImages):
    plt.savefig(fpath+"SurvivalFuncLog.svg", bbox_inches='tight')
plt.show()


#validate equation for retFrac
r = 0.05
i = 0.03
yR = 20 #years retired
retFrac(r,yR,i) #0.05966
retFrac(r,yR,0) #0.07642
fY = 0.06 #fraction to withdraw each year
p0 = 100 #initial stock price
c0 = 100 #initial CPI index
nY = np.linspace(0,yR-1,yR)
pN = p0*np.power(1+r,nY) #stock price each year
cN = c0*np.power(1+i,nY) #CPI each year

vR0 = 1e6 #value of investments at retirement
nS0 = vR0/pN[0] #number of stocks at retirement
w = vR0*retFrac(r,yR,0) #amount to withdraw every year
nSY = w/pN #number of stocks to sell each year

vRN = np.zeros(yR) #going to calculate value each year
vRN[0] = vR0-nSY[0]*pN[0] #withdraw at beginning of year
nSN = np.zeros(yR) #running tab on number of shares each year
nSN[0] = nS0-nSY[0] #number of shares at beginning of retirement
for y in range(1,yR):
    nSN[y] = nSN[y-1] - nSY[y]
    vRN[y] = vRN[y-1]*pN[y]/pN[y-1] - w

plt.figure()
plt.plot(nY,vRN/vR0)
plt.plot(nY,pN*nSN/vR0,'o')
plt.grid(True)
plt.ylim([0,1])
plt.xlim([0,yR])
plt.show

##########################################################################
###    Taxes
###    
##########################################################################

#plot marginal and effective tax rates for 2022
inc = 1000*np.linspace(10,400,100)
tbs = fn.taxBrackets('single')
tbj = fn.taxBrackets('joint')
trs = np.zeros(inc.size)
trj = np.zeros(inc.size)
for i in np.arange(inc.size):
    trs[i] = fn.taxRate(inc[i],tbs)
    trj[i] = fn.taxRate(inc[i],tbj)
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Marginal and Effective Tax Rates for 2022')
plt.step(tbs[:,0]/1000,100*tbs[:,1],'k--o',markersize=5,where='post',label='marg single')
plt.plot(inc/1000,100*trs,'k--',label='effect single')
plt.step(tbj[:,0]/1000,100*tbj[:,1],'k-s',markersize=5,where='post',label='marg joint')
plt.plot(inc/1000,100*trj,'k-',label='effect joint')
plt.grid(True)
plt.legend()
plt.ylim([9,36])
plt.xlim([0,inc[-1]/1000])
plt.xlabel('annual income ($k)')
plt.ylabel('effective tax rate (%)')
if (saveImages):
    plt.savefig(fpath+"TaxRate2022.svg", bbox_inches='tight')
plt.show()


#three cases:
#1:  taxed ordinary income going in, taxed capital gains coming out (unprotected)
#2:  posttax going in, untaxed coming out (roth)
#3:  pretax going in, taxed ordinary coming out (traditional)

Tw = 0.17 #ordinary income tax rate while working
Tr = 0.12 #ordinary income tax rate during retirement
Tc = 0.20 #capital gains tax rate

yS = 50 #number of years saving
y = np.linspace(0,yS,yS+1)
r = 0.08 #return rate
c = 5 #pretax contribution
i = 0.02 #inflation rate

plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Effective Savings after tax\nr={:.0f}%, c=${}k, i={:.0f}%'.format(r*100,c,i*100))
plt.plot(y,fn.taxedRetirement(r,c,y,i,0,Tr),'k:',label='traditional\nTw={:.2f}, Tr={:.2f}'.format(0,Tr))
plt.plot(y,fn.taxedRetirement(r,c,y,i,Tw,0),'k--',label='roth\nTw={:.2f}, Tr={:.2f}'.format(Tw,0))
plt.plot(y,fn.taxedRetirement(r,c,y,i,Tw,Tc),'k-',label='fully taxed\nTw={:.2f}, Tr={:.2f}'.format(Tw,Tc))
plt.grid(True)
plt.legend()
#plt.yscale('log')
#plt.ylim([1e-3,105])
plt.xlim([0,yS])
plt.xlabel('year')
plt.ylabel('effective savings ($k)')
#plt.text(2,410,'r={:.2f}, c={:.2f}, i={:.2f}'.format(r,c,i)) #, fontsize=annotateSize)
if (saveImages):
    plt.savefig(fpath+"EffectiveSavingsTax.svg", bbox_inches='tight')
plt.show()
