'''




'''
import numpy as np 

import catalog as Cat
import observables as Obvs

import matplotlib.pyplot as plt 
from ChangTools.plotting import prettycolors


def plotPureCentral_SHMF(nsnap_ancestor=20):
    ''' Plot the 
    '''
    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subcat = subhist.Read(downsampled='33')
    
    fig = plt.figure(1)
    sub = fig.add_subplot(111)

    for i in range(1, 21): 
        if i == 1: 
            masses = subcat['halo.m']
            weights = subcat['weights']
        else: 
            masses = subcat['snapshot'+str(i)+'_halo.m']
            weights = subcat['weights']

        shmf = Obvs.getMF(masses, weights=weights, m_arr=np.arange(10., 15.5, 0.1))
        sub.plot(shmf[0], shmf[1]) 

    # x-axis
    sub.set_xlim([10., 15.])
    # y-axis
    sub.set_yscale("log") 
    plt.show() 


def Test_nsnap_start(nsnap_ancestor=20):
    ''' 
    '''
    subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    subcat = subhist.Read(downsampled='33')
    
    pretty_colors = prettycolors()
    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    
    masses = subcat['m.star']
    weights = subcat['weights']

    for i in range(1, 21)[::-1]: 
        started = np.where(subcat['nsnap_start'] == i)

        mf = Obvs.getMF(masses[started], weights=weights[started])

        if i == 20: 
            mf_list = [np.zeros(len(mf[0]))]

        mf_list.append(mf[1] + mf_list[-1])

     
    for i in range(len(mf_list)-1):
        sub.fill_between(mf[0], mf_list[i], mf_list[i+1], edgecolor=None, color=pretty_colors[i]) 

    # x-axis
    sub.set_xlim([8., 12.])
    # y-axis
    sub.set_yscale("log") 
    plt.show() 



def test_Observations_GroupCat(): 
    ''' Simple test of the Cat.Observations class 
    '''
    real = Cat.Observations('group_catalog', Mrcut=18, position='central')
    catalog = real.Read()
    print catalog.keys()





if __name__=='__main__': 
    Test_nsnap_start(nsnap_ancestor=20)

    #test_Observations_GroupCat()
    #plotPureCentral_SHMF(nsnap_ancestor=20)

    #subhist = Cat.SubhaloHistory(nsnap_ancestor=20)
    #subhist.Build()
    #subhist._CheckHistory()
    #subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    #subhist.Build()
    #subhist.Downsample()
