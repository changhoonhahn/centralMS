''' 

Make figures for paper 


'''
import numpy as np 

import abcee as ABC

import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def SFHmodel():
    ''' Figure that illustrates the SFH of galaxies. 
    Two panel plot. Panel a) SFH of a galaxy plotted alongside SFMS 
    '''
    subcat = ABC.model('randomSFH', np.array([1.35, 0.6]), nsnap0=15, 
            downsampled='14', sigma_smhm=0.2)
    print subcat.key() 



if __name__=="__main__": 
    SFHmodel()
