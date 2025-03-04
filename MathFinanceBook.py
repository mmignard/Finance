# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 09:18:04 2023

@author: Marc
"""

# linestyles https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
# annotation https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html
# useful plot examples https://matplotlib.org/stable/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
#plt.annotate('local max', xy=(30,35), xytext=(35,25),
#             arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.text(30,45, r'$\mu=100,\ \sigma=15$')
# %matplotlib qt5 or %matplotlib inline switches from plot pane to separate window

from datetime import datetime
from datetime import date
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import finance as fn


#The line below creates graph titles and annotation text that is real text, which
#is potentially searchable. This renders correctly as a .svg image using Chrome
#and Edge, but when it is imported into MS Word, the font sizes are messed up.
#Removing this line turns all the text into graphic polygons so the file sizes
#are slightly larger, and the text is not searchable, but it looks correct in 
#MS Word. 
#plt.rcParams['svg.fonttype'] = 'none'
saveImages = False
annotateSize = 8
fpath = '..//images//'
##########################################################################
###    if you do not collect a return, calculate the amount you need to 
###    save each year for retirement 
##########################################################################
ageStartSaving = 20
yearsSaving = 50
yearsRetired = 20
yearOfWork = np.linspace(ageStartSaving,ageStartSaving+yearsSaving,1+yearsSaving)
ageStartRetirement = ageStartSaving+yearsSaving
yearOfRetirement = np.linspace(ageStartRetirement,ageStartRetirement+yearsRetired,1+yearsRetired)
grossIncome = 50 #1000 $ per year
savingsRate = 1/(1+yearsSaving/yearsRetired)
grossIncArr = np.ones(yearOfWork.size)*grossIncome
netIncArr = grossIncArr*(1-savingsRate)
retirementIncome = np.ones(yearOfRetirement.size)*grossIncome*(1-savingsRate)
workCumSavings = (yearOfWork - ageStartSaving)*grossIncome*(savingsRate)
savingsAtRetirement = workCumSavings[-1]
retirementCumSavings = savingsAtRetirement - ((yearOfRetirement - ageStartRetirement)*grossIncome*(1-savingsRate))

plt.figure(figsize=(5,3.5),dpi=300)
plt.subplot(211)
plt.title('Income and Savings, Ignoring Return')
plt.plot(yearOfWork,grossIncArr,'k')
plt.plot(yearOfWork,netIncArr,'k')
plt.plot(yearOfRetirement,retirementIncome,'k--')
plt.text(25,52,'gross income', fontsize=annotateSize)
plt.text(25,30,'net income', fontsize=annotateSize)
plt.text(78,23,'retirement\nincome', fontsize=annotateSize)
#text and double ended arrow
plt.text(50,42,'savings per year', fontsize=annotateSize)
plt.annotate('', xy=(48,50), xytext=(48,42), arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=8), fontsize=annotateSize)
plt.annotate('', xy=(48,36), xytext=(48,42), arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=8), fontsize=annotateSize)
plt.ylabel('$/1000')
plt.xlim([20,90])
plt.grid(True)
#do not plot x-axis tick labels on top plot because they just the same as the bottom plot
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.ylim([0,grossIncome+10])

plt.subplot(212)
plt.plot(yearOfWork,workCumSavings,'k')
plt.plot(yearOfRetirement,retirementCumSavings,'k--')
plt.xlabel('age')
plt.ylabel('$/1000')
plt.xlim([20,90])
plt.grid(True)
plt.ylim([0,1000])
plt.annotate('total savings you\nhave accumulated', xy=(45,360), xytext=(30,750), arrowprops=dict(facecolor='black', width=1, headwidth=5), fontsize=annotateSize)

if saveImages:
    plt.savefig(fpath+"retirementIncomeNoReturn.svg", bbox_inches='tight')
plt.show()

##########################################################################
###    if you do receive a return, calculate the new amount you need to 
###    save each year for retirement 
##########################################################################
r = 0.05 #100 year CAGR of DJIA
savingsRate = r*yearsRetired/(np.power(1+r,yearsSaving) - 1)
savingsRate = 1/(1+1/savingsRate)
netIncArr = grossIncArr*(1-savingsRate)
retirementIncome = np.ones(yearOfRetirement.size)*grossIncome*(1-savingsRate)
workCumSavings = grossIncome*savingsRate*((np.power(1+r,(yearOfWork - ageStartSaving)) - 1)/r)
savingsAtRetirement = workCumSavings[-1]
retirementCumSavings = savingsAtRetirement - ((yearOfRetirement - ageStartRetirement)*grossIncome*(1-savingsRate))
principleInvested = (yearOfWork - ageStartSaving)*grossIncome*(savingsRate)

plt.figure(figsize=(5,3.5),dpi=300)
plt.subplot(211)
plt.title('Income and Savings, With Return')
plt.plot(yearOfWork,grossIncArr,'k')
plt.plot(yearOfWork,netIncArr,'k')
plt.plot(yearOfRetirement,retirementIncome,'k--')
plt.text(25,52,'gross income', fontsize=annotateSize)
plt.text(25,40,'net income', fontsize=annotateSize)
plt.text(78,34,'retirement\nincome', fontsize=annotateSize)
#text and double ended arrow
plt.text(50,40,'savings per year', fontsize=annotateSize)
plt.annotate('', xy=(48,50), xytext=(48,58), arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=8), fontsize=annotateSize)
plt.annotate('', xy=(48,45), xytext=(48,37), arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=8), fontsize=annotateSize)
plt.ylabel('$/1000')
plt.xlim([20,90])
plt.grid(True)
#do not plot x-axis tick labels on top plot because they just the same as the bottom plot
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.ylim([0,grossIncome+10])

plt.subplot(212)
plt.plot(yearOfWork,workCumSavings,'k')
plt.plot(yearOfWork,principleInvested,'k:')
plt.plot(yearOfRetirement,retirementCumSavings,'k--')
plt.xlabel('age')
plt.ylabel('$/1000')
plt.grid(True)
plt.xlim([20,90])
plt.ylim([0,1000])
plt.annotate('total savings you\nhave accumulated', xy=(45,workCumSavings[25]), xytext=(30,750), arrowprops=dict(facecolor='black', width=1, headwidth=5), fontsize=annotateSize)
plt.annotate('principle\nactually\ninvested', xy=(60,principleInvested[40]), xytext=(65,350), arrowprops=dict(facecolor='black', width=1, headwidth=5), fontsize=annotateSize)

if saveImages:
    plt.savefig(fpath+"retirementIncomeWithReturn.svg", bbox_inches='tight')
plt.show()

##########################################################################
###    running cumulative savings with and without returns, assuming that
###    income is fixed at $40k/year during retirement 
##########################################################################

ageStartSaving = 20
yearsSaving = 50
yearsRetired = 20
ageStartRetirement = ageStartSaving+yearsSaving
grossIncome = 50 #1000 $ per year
retirementIncome = 40 #1000 $ per year
r = 0.05 #100 year CAGR of DJIA

1/(1+((1+r)**50-1)/(r*20))
50*(1-0.087)
savingsRate_Int = r*yearsRetired*retirementIncome/grossIncome/(np.power(1+r,yearsSaving)-1)
savingsRate_NoInt = yearsRetired*retirementIncome/grossIncome/yearsSaving

yearOfWork = np.linspace(ageStartSaving,ageStartSaving+yearsSaving,1+yearsSaving)
yearOfRetirement = np.linspace(ageStartRetirement,ageStartRetirement+yearsRetired,1+yearsRetired)
grossIncArr = np.ones(yearOfWork.size)*grossIncome
netIncArr_Int = grossIncArr*(1-savingsRate_Int)
netIncArr_NoInt = grossIncArr*(1-savingsRate_NoInt)
retirementIncomeArr = np.ones(yearOfRetirement.size)*retirementIncome
workCumSavings_Int = grossIncome*savingsRate_Int*((np.power(1+r,(yearOfWork - ageStartSaving)) - 1)/r)
workCumSavings_NoInt = (yearOfWork - ageStartSaving)*grossIncome*savingsRate_NoInt
savingsAtRetirement = workCumSavings_NoInt[-1]
retirementCumSavings = savingsAtRetirement - ((yearOfRetirement - ageStartRetirement)*retirementIncome)
principleInvested = (yearOfWork - ageStartSaving)*grossIncome*(savingsRate_Int)

plt.figure(figsize=(5,3.5),dpi=300)
plt.subplot(211)
plt.title('Income and Savings, Fixed Income')
plt.plot(yearOfWork,grossIncArr,'k')
plt.plot(yearOfWork,netIncArr_Int,'k')
plt.plot(yearOfWork,netIncArr_NoInt,'k')
plt.plot(yearOfRetirement,retirementIncomeArr,'k--')
plt.text(25,52,'gross income', fontsize=annotateSize)
plt.text(25,41,'net income with return', fontsize=annotateSize)
plt.text(25,29,'net income without return', fontsize=annotateSize)
plt.text(78,28,'retirement\nincome', fontsize=annotateSize)
plt.ylabel('$/1000')
plt.xlim([20,90])
plt.grid(True)
#do not plot x-axis tick labels on top plot because they just the same as the bottom plot
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.ylim([0,grossIncome+10])

plt.subplot(212)
plt.plot(yearOfWork,workCumSavings_Int,'k')
plt.plot(yearOfWork,workCumSavings_NoInt,'k')
plt.plot(yearOfWork,principleInvested,'k:')
plt.plot(yearOfRetirement,retirementCumSavings,'k--')
plt.xlabel('age')
plt.ylabel('$/1000')
plt.grid(True)
plt.xlim([20,90])
plt.ylim([0,1000])
plt.annotate('cumulative savings\nwith return', xy=(55,workCumSavings_Int[35]), xytext=(45,750), arrowprops=dict(facecolor='black', width=1, headwidth=5), fontsize=annotateSize)
plt.annotate('cumulative savings\nwithout return', xy=(38,290), xytext=(25,500), arrowprops=dict(facecolor='black', width=1, headwidth=5), fontsize=annotateSize)
plt.annotate('principle\nactually\ninvested', xy=(60,principleInvested[40]), xytext=(65,320), arrowprops=dict(facecolor='black', width=1, headwidth=5), fontsize=annotateSize)
if saveImages:
    plt.savefig(fpath+"fixedRetirementIncome.svg", bbox_inches='tight')
plt.show()
                          
##########################################################################
###     Smooth almost nowhere
###     
##########################################################################

# datYears = yf.download('SPY',interval = "3mo", start = datetime(2003,1,12), end = datetime(2023,3,8))
# datDays = yf.download('SPY',interval = "1d", start = datetime(2022,12,8,10), end = datetime(2023,3,8,10))
# datHours = yf.download('SPY',interval = "1h", start = datetime(2023,2,22), end = datetime(2023,3,8))
# datMinutes = yf.download('SPY',interval = "1m", start = datetime(2023,3,3,10), end = datetime(2023,3,3,11))

# datYears.to_csv('.//data//SpyYears.csv')
# datDays.to_csv('.//data//SpyDays.csv')
# datHours.to_csv('.//data//SpyHours.csv')
# datMinutes.to_csv('.//data//SpyMinutes.csv')

datYears = pd.read_csv('.//data//SpyYears.csv')
datDays = pd.read_csv('.//data//SpyDays.csv')
datHours = pd.read_csv('.//data//SpyHours.csv')
datMinutes = pd.read_csv('.//data//SpyMinutes.csv')

plt.figure(figsize=(5,5),dpi=300)
plt.subplots_adjust(wspace=0.25)
plt.subplot(221)
plt.title('Stock Prices Jump Around Over Every Timescale', loc='left')
t = np.zeros(datYears.shape[0])
for i in np.arange(datYears.shape[0]):
    dif = datYears.iloc[i].Date.date() - datYears.iloc[0].Date.date()
    temp = datetime.strptime(datYears.iloc[0].Date, '%Y-%m-%d')
    t[i] = dif.days/365 + dif.seconds/3600/24/365
plt.plot(t,datYears['Adj Close'],'k')
plt.text(1,150,'Years', fontsize=10)
plt.xlim([0,20])
plt.grid(True)

plt.subplot(222)
t = np.zeros(datDays.shape[0])
for i in np.arange(datDays.shape[0]):
    dif = datDays.iloc[i].name - datDays.iloc[0].name
    t[i] = dif.days + dif.seconds/3600/24
plt.plot(t,datDays['Adj Close'],'k')
plt.text(2,402,'Days', fontsize=10)
plt.xlim([0,60])
plt.grid(True)

plt.subplot(224)
t = np.zeros(datHours.shape[0])
for i in np.arange(datHours.shape[0]):
    dif = datHours.iloc[i].name - datHours.iloc[0].name
    t[i] = dif.days*24 + dif.seconds/3600
plt.plot(t,datHours['Adj Close'],'k')
plt.text(10,402,'Hours', fontsize=10)
plt.annotate('markets closed\non weekend', xy=(120,374), xytext=(150,365), arrowprops=dict(facecolor='black', width=1, headwidth=5), fontsize=annotateSize)

plt.xlim([0,300])
plt.grid(True)

plt.subplot(223)
t = np.zeros(datMinutes.shape[0])
for i in np.arange(datMinutes.shape[0]):
    dif = datMinutes.iloc[i].name - datMinutes.iloc[0].name
    t[i] = dif.days*24*60 + dif.seconds/60
plt.plot(t,datMinutes['Adj Close'],'k')
plt.text(2,401,'Minutes', fontsize=10)
plt.xlim([0,60])
#plt.ylim([391,394.01])
#plt.yticks(np.arange(391, 394.01, 1.0))
plt.grid(True)

if saveImages:
    plt.savefig(fpath+"StockPriceJumps.svg", bbox_inches='tight')
plt.show()
                    
##########################################################################
###    Show that average return rate is not a good measure. 
###    
##########################################################################

endTime = 49
time = 1 + np.linspace(0,endTime,1+endTime)
ret_flat = 0.05*np.ones(endTime+1)
ret_begLow = 0.1*np.ones(endTime+1)
ret_begLow[0:(int(endTime/2)+1)] = 0
ret_endLow = 0.1*np.ones(endTime+1)
ret_endLow[(int(endTime/2)+1):endTime+1] = 0

plt.figure(figsize=(5,3.5),dpi=300)
plt.subplot(211)
plt.title('Non-constant Rate of Return')
plt.plot(time,100*ret_flat,'k')
plt.plot(time,100*ret_begLow,'k--')
plt.plot(time,100*ret_endLow,'k:')
plt.ylabel('annual return (%)')
plt.grid(True)

#do not plot x-axis tick labels on top plot because they are just the same as the bottom plot
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

plt.subplot(212)
plt.plot(time,fn.accumReturn(ret_flat,c=1,i=0),'k',label='flat')
plt.plot(time,fn.accumReturn(ret_begLow,c=1,i=0),'k--',label='begin low')
plt.plot(time,fn.accumReturn(ret_endLow,c=1,i=0),'k:',label='end low')
plt.xlabel('year of saving')
plt.ylabel('savings ($)')
plt.grid(True)
plt.ylim([0,400])
plt.legend()
if saveImages:
    plt.savefig(fpath+"nonconstantReturnRate.svg", bbox_inches='tight')
plt.show()

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
###    Read and plot CPI data (inflation)
###    data from https://www.bls.gov/data/
##########################################################################

dat = np.genfromtxt('.//data//bls.gov//CPI_U_all_longTime.csv', dtype=float, delimiter=',', skip_header=12) 
year = dat[:,0]
cpiIndex = dat[:,1] #just use the number for January

plt.figure(figsize=(5,3.5),dpi=300)
plt.subplot(211)
plt.title('US Consumer Price Index')
plt.plot(year,cpiIndex,'k')
plt.ylabel('index (1983=100)')
#plt.yscale('log')
plt.grid(True)

#do not plot x-axis tick labels on top plot because they just the same as the bottom plot
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])

plt.subplot(212)
plt.plot(year[1:],100*(np.diff(cpiIndex)/cpiIndex[:-1]),'k')
plt.xlabel('year')
plt.ylabel('inflation (%)')
plt.grid(True)
#plt.ylim([0,400])
#plt.legend()
if saveImages:
    plt.savefig(fpath+"CPI_U_all_longTime.svg", bbox_inches='tight')
plt.show()

dat = np.genfromtxt('.//data//bls.gov//CPI_U_itemized.csv', dtype=float, delimiter=',', skip_header=12) 
plt.figure(figsize=(5,3.5),dpi=300)
plt.subplot()
plt.title('US Consumer Price Index Constituents')
plt.plot(dat[:,0],dat[:,6],'k:',linewidth=3,label='all')
#plt.plot(year,cpiIndex,'k:',linewidth=3,label='all')
plt.plot(dat[:,0],dat[:,1],'k',label='food')
plt.plot(dat[:,0],dat[:,2],'k--',label='shelter')
plt.plot(dat[:,0],dat[:,5],'k:',label='energy')
plt.plot(dat[:,0],dat[:,4],'k-.',label='medical')
#plt.yscale('log')
plt.xlabel('year')
#plt.xlim([1980,2025])
plt.ylabel('index (1983=100)')
plt.grid(True)
plt.legend()
if saveImages:
    plt.savefig(fpath+"CpiConstituents.svg", bbox_inches='tight')
plt.show()

##########################################################################
###     IAAGR-Inflation adjusted annual growth rate
###     
##########################################################################

r=0.1 #return rate (CAGR)
i=0.03 #inflation rate (CAIR)
c=5000 #annual investment
n = 50
year = np.linspace(1,n,30)
iaagr=(1+r)/(1+i)-1

doLogScale = False
cL = 100000
plt.figure(figsize=(5,3.5),dpi=300)
plt.subplot()
if (doLogScale):
    div = 1
    plt.title('Effects of inflation, logarithmic, CAGR={:.0f}%'.format(r*100))
else:
    div = 1e6
    plt.title('Effects of inflation, linear, CAGR={:.0f}%'.format(r*100))
i=0
iaagr=(1+r)/(1+i)-1
plt.plot(year,c/iaagr*(np.power(1+iaagr,year)-1)/div,'k',label='pmt, i={:.0f}%'.format(100*i))
i=0.02
iaagr=(1+r)/(1+i)-1
plt.plot(year,c/iaagr*(np.power(1+iaagr,year)-1)/div,'k--',label='pmt, i={:.0f}%'.format(100*i))
i=0.04
iaagr=(1+r)/(1+i)-1
plt.plot(year,c/iaagr*(np.power(1+iaagr,year)-1)/div,'k:',label='pmt, i={:.0f}%'.format(100*i))

i=0
plt.plot(year,cL*(np.power((1+r)/(1+i),year))/div,'ko',markersize=3,label='lump, i={:.0f}%'.format(100*i))
i=0.02
plt.plot(year,cL*(np.power((1+r)/(1+i),year))/div,'ko',markersize=1,label='lump, i={:.0f}%'.format(100*i))
i=0.04
plt.plot(year,cL*(np.power((1+r)/(1+i),year))/div,'kD',markersize=2,label='lump, i={:.0f}%'.format(100*i))

plt.grid(True)
if (doLogScale):
    plt.ylabel('cumulative account value, $')
    plt.yscale('log') 
    plt.ylim([c,12e6])
else:
    plt.ylabel('cumulative account value, $M')
    plt.ylim([0,12])

plt.xlabel('years')
plt.xlim([year[0],year[-1]])
plt.legend()
if (doLogScale):
    plt.annotate('one initial payment\nof ${:.0f}k'.format(cL/1000), xy=(3,1.5e5), xytext=(8,1.1e6), arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=8), fontsize=annotateSize)
    plt.annotate('annual payments\nof ${:.0f}k'.format(c/1000), xy=(2,1e4), xytext=(10,2e4), arrowprops=dict(facecolor='black', width=1, headwidth=5, headlength=8), fontsize=annotateSize)
else:
    plt.text(3,2.2,'There are two types\nof investments shown,\nOne initial payment of ${:.1f}k\nand annual payments of ${:.1f}k'.format(cL/1000,c/1000), fontsize=annotateSize)
if (saveImages):
    if (doLogScale):
        plt.savefig(fpath+"EffectsOfInflation_Log.svg", bbox_inches='tight')
    else:
        plt.savefig(fpath+"EffectsOfInflation.svg", bbox_inches='tight')
plt.show()

plt.figure(figsize=(5,1.75),dpi=300)
plt.subplot()
plt.title('Percent error using inflation adjustment\nr-i instead of (1+r)/(1+i) is incorrect,\nbut fairly close')
fracErr = 100*((1/(r-i)*(np.power(1+r-i,year)-1)/(1/iaagr*(np.power(1+iaagr,year)-1))) - 1)
plt.plot(year,fracErr,'k',label='annual payments')
fracErr = 100*(np.power(1+r-i,year)/np.power((1+r)/(1+i),year)-1)
plt.plot(year,fracErr,'k:',label='initial payment')
#plt.plot(year,cPmt*year,'k:',label='cumulative\nannual')
plt.ylabel('% error')
plt.xlabel('years')
#plt.yscale('log')
plt.xlim([year[0],year[-1]])
#plt.ylim([0,8])
plt.legend()
#plt.text(22,1000,'PMT=${}k, n={},r={}\ninital lump\ncontribution=${:.1f}k'.format(cPmt,n,r,cL), fontsize=annotateSize)
#plt.text(2,660,'inital lump contribution=${:.1f}k\nwith annual contribution=${}k,\ntotal contribution after {} years=${:.0f}k,\nCAGR={}%'.format(cL,cPmt,n,n*cPmt,100*r), fontsize=annotateSize)
plt.grid(True)
if (saveImages):
    plt.savefig(fpath+"ErrorRminusI.svg", bbox_inches='tight')
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

#quartT = np.floor(contT*4,dtype=float)/4
#quartInt = np.power(1+i/4,4)-1
#q = np.power(1+quartInt,quartT)

quartT = np.floor(contT*4,dtype=float)
q = np.power(1+i/4,quartT)

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
#import matplotlib.ticker as ticker
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
inc = 1000*np.linspace(1,400,100)
tbs = fn.taxBrackets('single')
tbj = fn.taxBrackets('joint')
trs = np.zeros(inc.size)
trj = np.zeros(inc.size)
for i in np.arange(inc.size):
    trs[i] = fn.taxRate(inc[i],tbs,fn.stDed('single'))
    trj[i] = fn.taxRate(inc[i],tbj,fn.stDed('joint'))
trs[trs<0] = 0
trj[trj<0] = 0
    
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Marginal and Effective Tax Rates for 2022\nassuming standard deduction')
plt.step(tbs[:,0]/1000,100*tbs[:,1],'k--o',markersize=5,where='post',label='marg single')
plt.plot(inc/1000,100*trs,'k--',label='effect single')
plt.step(tbj[:,0]/1000,100*tbj[:,1],'k-s',markersize=5,where='post',label='marg joint')
plt.plot(inc/1000,100*trj,'k-',label='effect joint')
plt.grid(True)
plt.legend()
plt.ylim([0,36])
plt.xlim([0,inc[-1]/1000])
plt.xlabel('annual income ($k)')
plt.ylabel('effective tax rate (%)')
if (saveImages):
    plt.savefig(fpath+"TaxRate2022.svg", bbox_inches='tight')
plt.show()

tbs = fn.taxBrackets('single')
50000*fn.taxRate(50000,tbs)
45000*fn.taxRate(45000,tbs)

#historical data on tax rates
tr = np.genfromtxt('.//data//taxfoundation.org//MaxTaxRate.csv',delimiter=',',skip_header=1,unpack=True,dtype=float)
trEff = np.genfromtxt('.//data//cbo.gov//HistoricalTaxRate.csv',delimiter=',',skip_header=1,unpack=True,dtype=float)
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Effective Tax Rate Versus Capital Gains Tax')
#plt.plot(tr[0,:],tr[1,:],'k--',label='max marginal')
plt.plot(tr[0,:],tr[2,:],'k-',label='capital gain')
plt.plot(trEff[0,:],trEff[4,:],'k--',label='eff rate top 1%')
plt.plot(trEff[0,:],trEff[3,:],'k-.',label='eff rate top 20%')
plt.plot(trEff[0,:],trEff[2,:],'k:',label='eff rate 20-80%')
plt.grid(True)
plt.legend(loc='lower left')
plt.ylim([5,41])
plt.xlim([1960,2020])
plt.xlabel('year')
plt.ylabel('maximum tax rate (%)')
if (saveImages):
    plt.savefig(fpath+"EffTaxVsCapGains.svg", bbox_inches='tight')
plt.show()

#three cases:
#1:  taxed ordinary income going in, taxed capital gains coming out (unprotected)
#2:  posttax going in, untaxed coming out (roth)
#3:  pretax going in, taxed ordinary coming out (traditional)

Tw = 0.22 #ordinary income tax rate while working (marginal)
Tr = 0.18 #ordinary income tax rate during retirement (average of marginal and effective)
Tc = 0.20 #capital gains tax rate (15% federal, 5% state)

yS = 50 #number of years saving
y = np.linspace(0,yS,yS+1)
r = 0.08 #return rate
c = 5 #pretax contribution
i = 0.02 #inflation rate

plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Effective Savings after tax\nr={:.0f}%, c=${}k, i={:.0f}%'.format(r*100,c,i*100))
plt.plot(y,(1-Tr)*fn.calcReturnPmt(r,c,y,i),'k:',label='traditional\nTw={:.2f}, Tr={:.2f}'.format(0,Tr))
plt.plot(y,(1-Tw)*fn.calcReturnPmt(r,c,y,i),'k--',label='roth\nTw={:.2f}, Tr={:.2f}'.format(Tw,0))
plt.plot(y,(1-Tw)*(1-Tc)*fn.calcReturnPmt(r,c,y,i)+Tc*(1-Tw)*y*c,'k-',label='fully taxed\nTw={:.2f}, Tc={:.2f}'.format(Tw,Tc))
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

#long term capital gains tax rate
def LTCGTR(MAGI,file='single'):  
    '''
    '''
    if ('single'==file):
        a=0


#net investment income tax

##########################################################################
###    Time saving for retirement
###    
##########################################################################

def reqRelSaving(r,n1,n2,i=0):
    return (np.power((1+r)/(1+i),n1+1)-1)/(np.power((1+r)/(1+i),n2+1)-1)
r=0.05
i=0
ageReference = 30
ageRetirement = 70 
age = np.linspace(20,50,31)
n1=ageRetirement-ageReference;
n2=ageRetirement-age
rrS = reqRelSaving(r,n1,n2,i)
    
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Required savings if starting before or\nafter {} years old, retirement age is {}'.format(ageReference,ageRetirement))
plt.plot(age,rrS,'k') #,label='req saving')
plt.grid(True)
#plt.legend()
plt.ylim([0.5,3.5])
plt.xlim([20,50])
plt.xlabel('age starting to save')
plt.ylabel('required relative savings')
for k in [10,5,-5,-10]:
    a = ageRetirement-(n1+k)
    rel = reqRelSaving(r,n1,n1+k,i)
    plt.plot([a,a+2.5],[rel,rel],'k:',linewidth=1)
    plt.text(a+2.8,rel-0.05,'{:0.2f}x'.format(rel), fontsize=annotateSize)
plt.annotate('reference age\nis {} years'.format(ageReference), xy=(ageReference,1), xytext=(ageReference-2,1.5), arrowprops=dict(facecolor='black', width=0.5, headwidth=5, headlength=8), fontsize=annotateSize)

if (saveImages):
    plt.savefig(fpath+"reqRelSaving.svg", bbox_inches='tight')
plt.show()

##########################################################################
###    Income increase with age
###    
##########################################################################
#from https://www.bls.gov/webapps/legacy/cpswktab3.htm, age & median weekly income
weeklyIncome = np.array(
                [[16,606],
                [20,706],
                [25,992],
                [35,1181],
                [45,1194],
                [55,1154],
                [65,1031],
                [80,1031]])

plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Age versus median annual income, all workers')
plt.step(weeklyIncome[:,0],weeklyIncome[:,1]*52/1000,'k',markersize=5,where='post',label='marg single')
plt.grid(True)
#plt.legend()
#plt.ylim([0.5,3.5])
plt.xlim([15,70])
plt.xlabel('age')
plt.ylabel('median annual income ($k)')

if (saveImages):
    plt.savefig(fpath+"AgeVsIncome.svg", bbox_inches='tight')
plt.show()

##########################################################################
###    Effect of management fees
###    
##########################################################################

def mangeFeesLump(r,m,t):
    """

    Parameters
    ----------
    r : float
        return rate.
    m : float
        management fees.
    t : float
        tax rate.

    Returns
    -------
    Number of years it takes for yearly management fees to be 
    greater than tax shelter benefits with a single lump sum payment
    at the beginning

    """
    return (np.log(1-t)/np.log((1+r-m)/(1+r)))

def mangeFeesPeriodEst(r,m,t):
    """

    Parameters
    ----------
    r : float
        return rate.
    m : float
        management fees.
    t : float
        tax rate.

    Returns
    -------
    Number of years it takes for yearly management fees to be 
    greater than tax shelter benefits with a periodic payments.
    This is not an exact solution, but the closest number of whole years.

    """
    n = np.zeros(m.shape[0])
    for mIdx in np.arange(m.shape[0]):
        n[mIdx]=10
        while((1-t)/r*(np.power(1+r,n[mIdx]+1)-1) < 1/(r-m[mIdx])*(np.power(1+r-m[mIdx],n[mIdx]+1)-1)):
        #while((1-t)*np.power(1+r,n[mIdx]+1) < np.power(1+r-m[mIdx],n[mIdx]+1)): #test against mangeFeesLump
            n[mIdx]=n[mIdx]+0.1
    return (n)

r=0.05
taxRate = 0.22
mFees = np.linspace(0.5,2,50)
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Years for management fees to exceed tax benefits\nreturn={}%, tax={}%'.format(100*r,taxRate*100))
plt.plot(mFees,mangeFeesPeriodEst(0.05,mFees/100,taxRate),'k',label='periodic')
plt.plot(mFees,mangeFeesLump(0.05,mFees/100,taxRate),'k--',label='lump')
plt.grid(True)
plt.legend()
plt.ylim([10,70])
plt.xlim([0.5,2])
plt.xlabel('management fees (%)')
plt.ylabel('number of years')

if (saveImages):
    plt.savefig(fpath+"ManagementFees.svg", bbox_inches='tight')
plt.show()

##########################################################################
###    Scatterplot of income
###    
##########################################################################
path = 'c://Users//mmig1//Documents//IPUMSdata//Attempt_2//cps_00002.dat'
imType = 'png'
col_specification = [(64,66), (66,75), (75,83)]
data = pd.read_fwf(path, colspecs=col_specification)
d_filt = data[data.iloc[:,0]>19]
d_filt2 = d_filt[d_filt.iloc[:,2]>2500]
d_filt3 = d_filt[d_filt.iloc[:,1]>2500]
sz = d_filt2.shape[0]

n=d_filt.shape[0]
d=d_filt.iloc[0:int(n/5),:] #too much data, so only look at some of it

#raw data, quantized on both age (horizontal) and income (vertical) axes
sprd_age = 0
sprd_inc = 0
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Annual income, all sources')
plt.plot(d.iloc[:,0]+sprd_age,(d.iloc[:,1]+sprd_inc)/1000,'ks',markersize=0.15)
plt.grid(True)
plt.ylim([1,200])
plt.xlim([20,90])
plt.xlabel('age')
plt.ylabel('Total annual income ($k)')
if saveImages:
    plt.savefig(fpath+"AgeVsIncTst01."+imType, bbox_inches='tight')
plt.show()

#randomize the age (horizonatal) axis by 1 year to remove quantization
sprd_inc = 0
sprd_age = np.random.uniform(low=0,high=0.999,size=d.shape[0])
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Annual income, all sources')
plt.plot(d.iloc[:,0]+sprd_age,(d.iloc[:,1]+sprd_inc)/1000,'ks',markersize=0.15)
plt.grid(True)
plt.ylim([1,200])
plt.xlim([20,90])
plt.xlabel('age')
plt.ylabel('Total annual income ($k)')
if saveImages:
    plt.savefig(fpath+"AgeVsIncTst02."+imType, bbox_inches='tight')
plt.show()

#some quantization on the income (vertical) axis, so randomize that too.
#increase the randomization to 5 years for ages 80 and 85, and lift incomes near 0 so they aren't obscured by axis
sprd_inc =  np.random.uniform(low=-2500,high=2500,size=d.shape[0])
sprd_age = np.random.uniform(low=0,high=0.999,size=d.shape[0])
idx = d.iloc[:,0]>=80 #find ages listed as 80 and 85
sprd_age[idx] = 5*sprd_age[idx] #and spread ages 80 and 85 over 5 years instead of 1 year
idx = d.iloc[:,1]<2500 #don't add random value to incomes of zero
sprd_inc[idx] = 2000 #but put them above lower axis so we can see them
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Annual income, all sources')
plt.plot(d.iloc[:,0]+sprd_age,(d.iloc[:,1]+sprd_inc)/1000,'ks',markersize=0.15)
plt.grid(True)
plt.ylim([1,200])
plt.xlim([20,90])
plt.xlabel('age')
plt.ylabel('Total annual income ($k)')
if saveImages:
    plt.savefig(fpath+"AgeVsIncTst03."+imType, bbox_inches='tight')
plt.show()

#logarithmic plot, remove incomes near 0
d2=d[d.iloc[:,1]>2500] #remove incomes near 0
sprd_inc =  np.random.uniform(low=-2500,high=2500,size=d2.shape[0])
sprd_age = np.random.uniform(low=0,high=0.999,size=d2.shape[0])
idx = d2.iloc[:,0]>=80
sprd_age[idx] = 5*sprd_age[idx]
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Annual income, all sources, logarithmic')
plt.plot(d2.iloc[:,0]+sprd_age,(d2.iloc[:,1]+sprd_inc),'ks',markersize=0.15)
plt.grid(True)
plt.ylim([1e3,1.18e6])
plt.yscale('log')
plt.xlim([20,90])
plt.xlabel('age')
plt.ylabel('Total annual income (k)')
if saveImages:
    plt.savefig(fpath+"AgeVsIncTst04."+imType, bbox_inches='tight')
plt.show()

#only look at wages, and remove other sources of income like retirement
d3=d[d.iloc[:,2]>2500] #remove incomes near 0, wages only
sprd_inc =  np.random.uniform(low=-2500,high=2500,size=d3.shape[0])
sprd_age = np.random.uniform(low=0,high=0.999,size=d3.shape[0])
idx = d3.iloc[:,0]>=80
sprd_age[idx] = 5*sprd_age[idx]
plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Annual income, wages only, logarithmic')
plt.plot(d3.iloc[:,0]+sprd_age,(d3.iloc[:,2]+sprd_inc),'ks',markersize=0.15)
plt.grid(True)
plt.ylim([1e3,1.18e6])
plt.yscale('log')
plt.xlim([20,90])
plt.xlabel('age')
plt.ylabel('Annual wages (k)')
if saveImages:
    plt.savefig(fpath+"AgeVsIncTst05."+imType, bbox_inches='tight')
plt.show()

##########################################################################
###    Population wave
###    
##########################################################################
#data from https://www.mortality.org/Country/Country?cntr=USA
dat = np.genfromtxt('.//data//AgeVsPopulationVsDate.csv', dtype=float, delimiter=',', skip_header=3) 

#estimated population in 2060 from https://www.statista.com/statistics/611644/united-states-population-projection-by-age/
popEst2060 = np.array(
    [[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
     [22778,22894,22871,23067,23999,25065,25845,26151,25949,25368,25395,24893,24357,24112,21662,18393,14274,10184,6184,2752,604]]
    )

year = dat[:,0]
cpiIndex = dat[:,1] #just use the number for January
2021-1932
111
pop = np.reshape(dat,(89,111,-1))
#first index is year, 1933+i (i in 0:89, 1933-2021)
#second index is age, j (0-110)
for i in np.arange(0,32,1):
    print(i,pop[i,110,:])

plt.figure(figsize=(5,3.5),dpi=300)
plt.title('Change in total US population')
#for i in [7,27,47,67,87]:
#    plt.plot(pop[i,:,1],pop[i,:,4]/1e6,label='{}'.format(i+1933))
plt.plot(popEst2060[0,:],popEst2060[1,:]/5e3,'k-o',markersize=3,label='{}'.format(2060))
i=87; plt.plot(pop[i,:,1],pop[i,:,4]/1e6,'k',label='{}'.format(i+1933))
i=47; plt.plot(pop[i,:,1],pop[i,:,4]/1e6,'k--',label='{}'.format(i+1933))
i=7; plt.plot(pop[i,:,1],pop[i,:,4]/1e6,'k:',label='{}'.format(i+1933))
plt.grid(True)
plt.ylim([0,6])
#plt.yscale('log')
plt.xlim([0,100])
plt.xlabel('age')
plt.ylabel('population in millions')
plt.legend()

if saveImages:
    plt.savefig(fpath+"PopulationAgeChange.svg", bbox_inches='tight')
plt.show()

