'''
James Douglass
20140216

Use Pandas and Matplotlib to generate summary of CSV

'''

import numpy as np
import pandas as pd
import pylab as pl
from scipy.optimize import curve_fit

filename = 'test1.csv'

def csv2df(filename):
    '''takes file and returns pandas dataframe'''
    df = pd.read_csv(filename)
    return df

def csv2np(filename):
    ''' takes file and return numpy array'''
    data = np.recfromcsv(filename)
    return data

##------ PLOT HELPER --------##

def plot_settings_scatter():
    #Move Splines
    ax = pl.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    #ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    #ax.spines['left'].set_position(('data',0))
    #Add Legend

def line_fit(x, y):
    def fit_func(x, a, b):
        return (a*x + b)
    params = curve_fit(fit_func, x, y)
    [a, b] = params[0]
    return [a, b]

def apply_f(x, a, b):
    y = []
    for i in range(len(x)):
        y.append(a*x[i] + b)
    return y

##------ PLOT FUNCTIONS --------##  

def plot_hist(df, field):
    '''plots histagram from pandas dataframe'''
    save_name = 'hist_' + field + '.png'
    pl.figure()
    df[field].diff().hist()
    pl.savefig(save_name)

def plot_scatter(df, field1, field2):
    '''plots scatterplot from pandas datafram'''
    save_name = 'scatter_' + field1 + '_' + field2 +'.png'
    x = df[field1]
    y = df[field2]
    pl.figure(figsize=(16,9), dpi=100)
    pl.subplot(1,1,1)
    plot_settings_scatter()
    pl.plot(x, y, 'o')
    pl.legend(loc='upper left')
    pl.savefig(save_name)

def plot_scatter_line(df, field1, field2):
    '''plots scatterplot with line from pandas datafram'''
    save_name = 'scatter_' + field1 + '_' + field2 +'.png'
    x = df[field1]
    y = df[field2]
    [a,b] = line_fit(x, y)
    x2 = np.linspace(min(x), max(x), len(x))
    y2 = apply_f(x2, a , b)
    pl.figure(figsize=(16,9), dpi=100)
    pl.subplot(1,1,1)
    plot_settings_scatter()
    pl.plot(x, y, 'o')
    pl.plot(x2, y2)
    pl.legend(loc='upper left')
    pl.savefig(save_name)   

def plot_all_hist(df):
    pl.figure()
    df.diff().hist(color='k', alpha=0.5, bins=50)
    pl.show()

##------ MAIN --------##

df = []

def main(filename):
    global df
    df = csv2df(filename)
    return df

main()
