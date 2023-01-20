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

def simpson38(f, edges, bw, *arg):
    
    yedges = f(edges, *arg)
    left38 = f((2.*edges[1:]+edges[:-1]) / 3., *arg)
    right38 = f((edges[1:]+2.*edges[:-1]) / 3., *arg)
    return bw / 8.*( np.sum(yedges)*2.+np.sum(left38+right38)*3. - (yedges[0]+yedges[-1]) ) #simpson3/8


def integrate1d(f, bound, nint, *arg):
    """
    compute 1d integral
    """
    edges = np.linspace(bound[0], bound[1], nint+1)
    bw = edges[1] - edges[0]
    
    return simpson38(f, edges, bw, *arg)

class Datahandler():
    def __init__(self, data, bins = 10, sy = None, binneddata = True, badvalue=-100000):
        self.data = data
        self.bins = bins
        self.binned = binneddata
        self.badvalue = badvalue
        if binneddata == True:
            self.xrange = (min(data), max(data))
            y, edges = np.histogram(data, bins, range = self.xrange)
            x = (edges[1:] + edges[:-1])/2
            mask = y > 0
            self.y = y[mask]
            self.x = x[mask]
            self.binwidth = (self.xrange[1] - self.xrange[0])/self.bins
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
            self.binwidth = 1
        

        self.lastfittypes = None
        self.nfits = 0
        self.plotinit = False
        
    def basicinfo(self):
        if self.binned:
            entries = len(self.data)
            mean = np.mean(self.data)
            std = np.std(self.data, ddof = 1)
            eom = std/np.sqrt(entries)
            print(f"Data has {entries:d} entries with the mean = {mean:.3f} +/- {eom:.3f} and std = {std:.3f}")
            return entries, mean, std, eom

    def chi2fit(self, fitfunc, fulldata = False, **kwargs):

        def obt(*args):
            yvalues = fitfunc(self.x, *args)
            return np.sum((self.y - yvalues)**2/self.sy**2)

        chi2_Min = Minuit(obt, **kwargs, name = [*kwargs])
        chi2_Min.errordef = 1
        chi2_Min.migrad()
        chi2 = chi2_Min.fval
        Ndof = len(self.x) - chi2_Min.nfit
        prob = stats.chi2.sf(chi2, Ndof)
        self.fitfunc = fitfunc
        self.chi2valuefit = chi2_Min.fval
        self.Ndoffit = Ndof
        self.probfit = prob
        self.valuesfit = np.array(chi2_Min.values, dtype = np.float64)
        self.errorsfit = np.array(chi2_Min.errors, dtype = np.float64)
        self.lastfittype = "Chi2"
        self.nfits += 1
        self.fitparams = [*kwargs]
        if not chi2_Min.valid:
            print("!!! Fit did not converge !!!\n!!! Give better initial parameters !!!")
        if fulldata == True:
            return self.valuesfit, self.errorsfit, chi2, Ndof, prob
        elif not fulldata:
            return self.valuesfit, self.errorsfit

    def ullhfit(self, fitfunc, extended = True, extended_nint = 100, **kwargs):

        def obt(*args):
            # !!! Ripped directly from Troels Petersen's work !!!
            logf = np.zeros_like(self.data)
            
            # compute the function value
            f = fitfunc(self.data, *args)
        
            # find where the PDF is 0 or negative (unphysical)        
            mask_f_positive = f > 0

            # calculate the log of f everyhere where f is positive
            logf[mask_f_positive] = np.log(f[mask_f_positive])
            # set everywhere else to badvalue
            logf[~mask_f_positive] = self.badvalue
            
            # compute the sum of the log values: the LLH
            llh = -np.sum(logf)
            if extended:
                extended_term = integrate1d(fitfunc, self.xrange, extended_nint, *args)
                llh += extended_term
            return llh

        ullh_Min = Minuit(obt, **kwargs, name = [*kwargs])
        ullh_Min.errordef = 0.5
        ullh_Min.migrad()
        self.fitfunc = fitfunc
        self.valuesfit = np.array(ullh_Min.values, dtype = np.float64)
        self.errorsfit = np.array(ullh_Min.errors, dtype = np.float64)
        self.lastfittype = "ullh"
        self.nfits += 1
        self.fitparams = [*kwargs]
        if not ullh_Min.valid:
            print("!!! Fit did not converge !!!\n!!! Give better initial parameters !!!")
        # *** Impliment p-value for ullh fit
        return self.valuesfit, self.errorsfit

    
    def initplot(self, figsize = (10,7)):
        # Only singleplot functionallity
        self.fig, self.ax = plt.subplots(1,1, figsize = figsize)
        self.plotinit = True
            
    def quickplot(self, lineplot = True, errorplot = True, label = "Default", xlabel = "", ylabel = "", capsize = 3):
        if self.plotinit:
            if self.binned:
                self.ax.hist(self.data, bins = self.bins, histtype = "step", color = "b", label = label)
                if errorplot:
                    self.ax.errorbar(self.x, self.y, yerr = self.sy, fmt = ".", ecolor = "k", markersize = 0, capsize = capsize)
                self.ax.set_xlabel(xlabel, fontsize = 10)
                self.ax.set_ylabel(ylabel, fontsize = 10)            
            else:
                if lineplot:
                    self.ax.plot(self.x, self.y, color = "b", label = label)
                if errorplot:
                    self.ax.errorbar(self.x, self.y, yerr = self.sy, fmt = ".", 
                                     ecolor = "k", markersize = 7, color = "b", capsize = capsize, label = label)
                self.ax.set_xlabel(xlabel, fontsize = 10)
                self.ax.set_ylabel(ylabel, fontsize = 10)
        else:
            raise ValueError("No plot initialized. Do initplot()")

    def plotfit(self, xpos = 0.02, ypos = 0.98, extra_spacing=-3, decimals=3, N_plotpoints = 10000, topspace = 0.3, textboxspace = 1/3, **kwargs):
        if self.plotinit:
            if self.lastfittype == None:
                raise ValueError("!!! No fit has been done !!!\nPlease call a fit function before plotting")
            if self.lastfittype == "Chi2":
                xvalues = np.linspace(*self.xrange, N_plotpoints)
                yvalues = self.fitfunc(xvalues, *self.valuesfit)
                self.ax.plot(xvalues, yvalues, color = "r", label = "Chi2 fit")
                d_title = {"Chi2 fit": ""}
                d_params = {name: [value, error] for name, value, error in zip(self.fitparams, self.valuesfit, self.errorsfit)}
                d_rest = {
                    "Chi2": self.chi2valuefit,
                    "Ndof": self.Ndoffit,
                    "Prob": self.probfit
                }
                d = {**d_title, **d_params, **d_rest}
                text = nice_string_output(d, extra_spacing=extra_spacing, decimals=decimals)
                add_text_to_ax(xpos + textboxspace * (self.nfits-1), ypos, text, self.ax, **kwargs)
                ymin, ymax = self.ax.get_ylim()
                ymax = ymax + (ymax-ymin)*topspace
                self.ax.set_ylim(ymin, ymax)
            if self.lastfittype == "ullh":
                xvalues = np.linspace(*self.xrange, N_plotpoints)
                yvalues = self.fitfunc(xvalues, *self.valuesfit) * self.binwidth
                self.ax.plot(xvalues, yvalues, color = "g", label = "ULLH fit")
                d_title = {"ULLH fit": ""}
                d_params = {name: [value, error] for name, value, error in zip(self.fitparams, self.valuesfit, self.errorsfit)}
                d = {**d_title, **d_params}
                text = nice_string_output(d, extra_spacing=extra_spacing, decimals=decimals)
                add_text_to_ax(xpos + textboxspace * (self.nfits-1), ypos, text, self.ax, **kwargs)
                ymin, ymax = self.ax.get_ylim()
                ymax = ymax + (ymax-ymin)*topspace
                self.ax.set_ylim(ymin, ymax)
        else:
            raise ValueError("No plot initialized. Do initplot()")

    def legend(self):
        self.ax.legend(frameon = False, fontsize = 12, loc = "upper right")

    def savefig(self, name = "Default", dpi = 400):
        self.fig.tight_layout()
        self.fig.savefig(name, dpi = dpi)