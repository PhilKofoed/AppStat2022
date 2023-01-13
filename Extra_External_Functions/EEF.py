# coding: utf-8

#Author: Christian Michelsen, NBI, 2018
#        Troels Petersen, NBI, 2019-22
#        Philip Kofoed-Djursner, KU, 22-23

import numpy as np
import matplotlib.pyplot as plt                        # Plots and figures like you know them from Matlab                             # Make the plots nicer to look at
from iminuit import Minuit
from scipy import stats     

def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float64)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.int64)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-1]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k', halignment = "left", valignment = "top"):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, va=valignment, ha = halignment, color=color)
    return None

class Datahandler():
    def __init__(self, data, bins = 10, sy = None, binneddata = True):
        self.data = data
        self.bins = bins
        self.binned = binneddata
        if binneddata == True:
            self.xrange = (min(data), max(data))
            y, edges = np.histogram(data, bins, range = self.xrange)
            x = (edges[1:] + edges[:-1])/2
            mask = y > 0
            self.y = y[mask]
            self.x = x[mask]
            if sy == None:
                sy_poi = np.sqrt(y)
                self.sy = sy_poi[mask]
            else:
                self.sy = sy
        else:
            self.x = np.array(data[0])
            self.y = np.array(data[1])
            self.sy = np.array(data[2])
            self.xrange = (min(self.x), max(self.x))
        self.lastfittype = None
        self.plotinit = False
        
    def basicinfo(self):
        if self.binned:
            entries = len(self.data)
            mean = np.mean(self.data)
            std = np.std(self.data, ddof = 1)
            eom = std/np.sqrt(entries)
            print(f"Data has {entries:d} entries with the mean = {mean:.3f} +/- {eom:.3f} and std = {std:.3f}")
            return entries, mean, std, eom

    def chi2fit(self, func, fulldata = False, **kwargs):

        def obt(*args):
            yvalues = func(self.x, *args)
            return np.sum((self.y - yvalues)**2/self.sy**2)

        chi2_Min = Minuit(obt, **kwargs, name = [*kwargs])
        chi2_Min.errordef = 1
        chi2_Min.migrad()
        chi2 = chi2_Min.fval
        Ndof = len(self.x) - chi2_Min.nfit
        prob = stats.chi2.sf(chi2, Ndof)
        self.fitfunc = func
        self.chi2valuefit = chi2_Min.fval
        self.Ndoffit = Ndof
        self.probfit = prob
        self.valuesfit = np.array(chi2_Min.values, dtype = np.float64)
        self.errorsfit = np.array(chi2_Min.errors, dtype = np.float64)
        self.lastfittype = "Chi2"
        self.fitparams = [*kwargs]
        if not chi2_Min.valid:
            print("!!! Fit did not converge !!!\n!!! Give better initial parameters !!!")
        if fulldata == True:
            return self.valuesfit, self.errorsfit, chi2, Ndof, prob
        elif not fulldata:
            return self.valuesfit, self.errorsfit

    def initplot(self, size = (10,7)):
        # Only singleplot functionallity
        self.fig, self.ax = plt.subplots(1,1, figsize = size)
        self.plotinit = True
            
    def quickplot(self):
        if self.plotinit:
            if self.binned:
                self.ax.hist(self.data, bins = self.bins, histtype = "step", color = "b")
                self.ax.errorbar(self.x, self.y, yerr = self.sy, fmt = ".", ecolor = "k", markersize = 0, capsize = 3)
            else:
                self.ax.plot(self.x, self.y, color = "b")
                self.ax.errorbar(self.x, self.y, yerr = self.sy, fmt = ".", ecolor = "k", markersize = 0, capsize = 3)
        else:
            raise ValueError("No plot initialized. Do initplot()")

    def plotfit(self, xpos = 0.02, ypos = 0.98, extra_spacing=5, decimals=3, N_plotpoints = 10000, **kwargs):
        if self.plotinit:
            if self.lastfittype == None:
                raise ValueError("!!! No fit has been done !!!\nPlease call a fit function before plotting")
            if self.lastfittype == "Chi2":
                xvalues = np.linspace(*self.xrange, N_plotpoints)
                yvalues = self.fitfunc(xvalues, *self.valuesfit)
                plt.plot(xvalues, yvalues, color = "r")
                d_params = {name: [value, error] for name, value, error in zip(self.fitparams, self.valuesfit, self.errorsfit)}
                d_rest = {
                    "Chi2": self.chi2valuefit,
                    "Ndof": self.Ndoffit,
                    "Prob": self.probfit
                }
                d = {**d_params, **d_rest}
                text = nice_string_output(d, extra_spacing=extra_spacing, decimals=decimals)
                add_text_to_ax(xpos, ypos, text, self.ax, **kwargs)
                ymin, ymax = self.ax.get_ylim()
                ymax = ymax + (ymax-ymin)*0.3
                self.ax.set_ylim(ymin, ymax)
        else:
            raise ValueError("No plot initialized. Do initplot()")