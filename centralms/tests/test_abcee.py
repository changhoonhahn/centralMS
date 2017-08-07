'''


'''
import time
import numpy as np 

# -- local -- 
import env 
import abcee
import models
import util as UT

# --- plotting --- 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def test_SumData(): 
    ''' Make sure abcee.SumData returns something sensible with some hardcoded values 
     
    Takes roughly 0.7 seconds 
    '''
    t0 = time.time() 
    output = abcee.SumData(['smf'], nsnap0=15, downsampled='14') 
    print time.time() - t0 , ' seconds'
    return output


def test_SumSim():
    ''' Profile the simulation 

    Takes roughly ~5 seconds for "constant offset" 
    '''
    t0 = time.time() 
    # run the model 
    subcat = models.model('test0', np.array([1.05, 0.53]), nsnap0=15, downsampled='14')
    # get summary statistics 
    output = abcee.SumSim(['smf'], subcat)
    print time.time() - t0, ' seconds'

    return output 


def test_Dist(): 
    ''' Test distance metric 
    '''
    # data summary statistic
    sum_data = test_SumData()

    # simulation summary statistic 
    sum_sims = test_SumSim()

    # distance function  
    rho = abcee.roe_wrap(['smf'])  
    
    # calculate distance 
    d = rho(sum_sims, sum_data)
    
    return d 


def test_runABC(): 
    ''' Purely a test run to check that there aren't any errors 
        
    ################################
    Run successful!
    ################################
    '''
    abcee.runABC('test0', 2, [10.], N_p=5, sumstat=['smf'], nsnap0=15, downsampled='14')
    return None 


def test_readABC(T): 
    ''' Try reading in different ABC outputs and do basic plots 
    '''
    abcout = abcee.readABC('test0', T)
    
    # median theta 
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]

    print theta_med


def test_ABCsumstat(run, T):#, sumstat=['smf']): 
    ''' Compare the summary statistics of the median T-th ABC particle pool with data.
    Hardcoded for smf only 
    '''
    # data summary statistic
    sumdata = abcee.SumData(['smf'], info=True, nsnap0=15)

    # median theta 
    abcout = abcee.readABC('test0', T)
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
    
    subcat = models.model(run, theta_med, nsnap0=15, downsampled='14')
    sumsim = abcee.SumSim(['smf'], subcat, info=True)

    fig = plt.figure()
    sub = fig.add_subplot(111)

    sub.plot(sumdata[0][0], sumdata[0][1], c='k', ls='--', label='Data')
    sub.plot(sumsim[0][0], sumsim[0][1], c='b', label='Sim.')

    sub.set_xlim([6., 12.])
    sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
    sub.set_ylim([1e-6, 10**-1.75])
    sub.set_yscale('log')
    sub.set_ylabel('$\Phi$', fontsize=25)
    sub.legend(loc='upper right') 
    plt.show()

    return None 


def test_ABC_SMHMR(run, T):#, sumstat=['smf']): 
    ''' Compare the summary statistics of the median T-th ABC particle pool with data.
    Hardcoded for smf only 
    '''
    # data summary statistic
    subcat_dat = abcee.Data(nsnap0=15)

    # median theta 
    abcout = abcee.readABC('test0', T)
    theta_med = [UT.median(abcout['theta'][:, i], weights=abcout['w'][:]) for i in range(len(abcout['theta'][0]))]
    
    subcat_sim = models.model(run, theta_med, nsnap0=15, downsampled='14')

    fig = plt.figure()
    sub = fig.add_subplot(111)

    sub.plot(sumdata[0][0], sumdata[0][1], c='k', ls='--', label='Data')
    sub.plot(sumsim[0][0], sumsim[0][1], c='b', label='Sim.')

    sub.set_xlim([6., 12.])
    sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
    sub.set_ylim([1e-6, 10**-1.75])
    sub.set_yscale('log')
    sub.set_ylabel('$\Phi$', fontsize=25)
    sub.legend(loc='upper right') 
    plt.show()

    return None 


if __name__=='__main__': 
    #print test_SumData()
    #test_runABC()
    test_ABCsumstat('test0', 9)
    #for t in range(10)[::-1]: 
    #    test_readABC(t)
