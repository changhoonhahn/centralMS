'''




'''
import numpy as np 

import env
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


def plotPureCentral_SMHMR(sigma_smhm=0.2, nsnap0=15): 
    subhist = Cat.PureCentralHistory(sigma_smhm=sigma_smhm, nsnap_ancestor=nsnap0)
    subcat = subhist.Read()
    
    smhmr = Obvs.Smhmr()
    m_mid, mu_mstar, sig_mstar, cnts = smhmr.Calculate(subcat['m.max'], subcat['m.star'])
    #plt.errorbar(m_mid, mu_mstar, yerr=sig_mstar) 
    plt.plot(m_mid, sig_mstar, c='k') 

    m_mid, mu_mstar, sig_mstar, cnts = smhmr.Calculate(subcat['snapshot'+str(nsnap0)+'_m.max'], 
            subcat['snapshot'+str(nsnap0)+'_m.star'])
    plt.plot(m_mid, sig_mstar) 
    plt.ylim([0., 0.5]) 
    print subcat['snapshot'+str(nsnap0)+'_m.max'].min() 
    print subcat['snapshot'+str(nsnap0)+'_m.star'][subcat['snapshot'+str(nsnap0)+'_m.star'] > 0.].min() 


    weird = subcat['snapshot'+str(nsnap0)+'_m.star'][subcat['snapshot'+str(nsnap0)+'_m.max']<11.25]
    print weird.min(), weird.max() 
    weird = subcat['snapshot'+str(nsnap0)+'_m.star'][subcat['snapshot'+str(nsnap0)+'_m.max']<10.01]
    print weird.min(), weird.max() 

    #plt.errorbar(m_mid, mu_mstar, yerr=sig_mstar) 
    plt.show() 
    return None 


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
    plotPureCentral_SMHMR(sigma_smhm=0.2, nsnap0=15)
    #Test_nsnap_start(nsnap_ancestor=20)

    #test_Observations_GroupCat()
    #plotPureCentral_SHMF(nsnap_ancestor=20)

    #test_Downsample('smhmr', 15)

    #for sig in [0.2, 0.]: 
    #    #subhist = Cat.Subhalos(sigma_smhm=sig, nsnap0=15)
    #    #subhist.Build()
    #    subhist = Cat.PureCentralSubhalos(sigma_smhm=sig, nsnap0=15)
    #    subhist.Build()
    #    subhist.Downsample()

    #subhist = Cat.SubhaloHistory(nsnap_ancestor=20)
    #subhist.Build()
    #subhist._CheckHistory()
    #subhist = Cat.PureCentralHistory(nsnap_ancestor=20)
    #subhist.Build()
    #subhist.Downsample()
