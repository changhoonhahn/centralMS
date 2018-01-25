'''

Test for observables


'''
import numpy as np 
from letstalkaboutquench.fstarforms import fstarforms

import env 
import util as UT
import catalog as Cat
import observables as Obvs

import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def SFMS_z1(): 
    ''' Comparison of different SFMS observations at z~1.
    '''
    # Lee et al. (2015)
    logSFR_lee = lambda mm: 1.53 - np.log10(1 + (10**mm/10**(10.10))**-1.26)
    # Noeske et al. (2007) 0.85 < z< 1.10 (by eye)
    logSFR_noeske = lambda mm: (1.580 - 1.064)/(11.229 - 10.029)*(mm - 10.0285) + 1.0637
    # Moustakas et al. (2013) 0.8 < z < 1. (by eye)  
    logSFR_primus = lambda mm: (1.3320 - 1.296)/(10.49 - 9.555) * (mm-9.555) + 1.297
    
    # prior range
    logSFR_prior_min = lambda mm: (0.6 * (mm - 10.5) + 0.9 * (1.-0.0502) - 0.11) 
    logSFR_prior_max = lambda mm: (0.6 * (mm - 10.5) + 2. * (1.-0.0502) - 0.11) 


    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    marr = np.linspace(9., 12., 100)
    sub.plot(marr, logSFR_lee(marr), label='Lee et al.(2015)')
    sub.plot(marr, logSFR_noeske(marr), label='Noeske et al.(2007)')
    sub.plot(marr, logSFR_primus(marr), label='Moustakas et al.(2013)')
    sub.fill_between(marr, logSFR_prior_min(marr), logSFR_prior_max(marr), label='Prior', alpha=0.5)
    sub.legend(loc='lower right') 
    sub.set_xlim([9.5, 11.5])
    fig.savefig(UT.fig_dir()+'test.z_1.sfms.png')
    plt.close() 
    return None


def SFMS_comparison(): 
    ''' Comparison of different SFMS observations at z~0.05 and 1. We 
    include the compilation fit from Speagle et al. (2014).  
    '''
    # z = 0 
    # fit from SDSS group catalog
    logSFR_sdss = lambda mm: (0.5757 * (mm - 10.5) - 0.13868)

    # Lee et al. (2015)
    logSFR_lee = lambda mm: 1.53 - np.log10(1 + (10**mm/10**(10.10))**-1.26)
    # Noeske et al. (2007) 0.85 < z< 1.10 (by eye)
    logSFR_noeske = lambda mm: (1.580 - 1.064)/(11.229 - 10.029)*(mm - 10.0285) + 1.0637
    # Moustakas et al. (2013) 0.8 < z < 1. (by eye)  
    logSFR_primus = lambda mm: (1.3320 - 1.296)/(10.49 - 9.555) * (mm-9.555) + 1.297

    # Speagle's SFMS bestfit
    logSFR_speagle = lambda mm, tt: (0.84 - 0.026 * tt) * mm - (6.51 - 0.11 * tt) 
    
    # priors 
    logSFR_prior = lambda mm, zz, p1, p2: logSFR_sdss(mm) + (zz - 0.05) * (p1 * (mm - 10.5) + p2)  
    marr = np.linspace(9., 12., 100)

    fig = plt.figure(1)
    sub = fig.add_subplot(121) # z = 0.05 subplot 
    sub.plot(marr, logSFR_sdss(marr), label='SDSS group catalog')  
    sub.plot(marr, logSFR_speagle(marr, 13.5), label='Speagle $z \sim 0.05$')  
    sub.legend(loc='lower right') 
    sub.set_xlim([9.5, 11.5])
    sub.set_ylim([-2, 3])
    
    sub = fig.add_subplot(122) # z = 1 subplot 
    sub.plot(marr, logSFR_lee(marr), label='Lee et al.(2015)')
    sub.plot(marr, logSFR_noeske(marr), label='Noeske et al.(2007)')
    sub.plot(marr, logSFR_primus(marr), label='Moustakas et al.(2013)')
    sub.plot(marr, logSFR_speagle(marr, 5.7), label='Speagle $z \sim 1$')  

    logSFRmin, logSFRmax = [], [] 
    parr = [[-0.5, 1], [-0.5, 2.], [0.5, 1.], [0.5, 2.]]
    for mm in marr: 
        logSFRmin.append(
                np.min([logSFR_prior(mm, 1., pp[0], pp[1]) for pp in parr]))
        logSFRmax.append(
                np.max([logSFR_prior(mm, 1., pp[0], pp[1]) for pp in parr]))

    sub.fill_between(marr, logSFRmin, logSFRmax, label='Prior', alpha=0.5)
    sub.legend(loc='lower right') 
    sub.set_xlim([9.5, 11.5])
    sub.set_ylim([-2, 3])
    fig.savefig(UT.fig_dir()+'test.sfms.comparison.png')
    plt.close() 
    return None


def SFMS_Mfid_z(): 
    ''' Comparison of different SFMS observations from z~0.05 to 1 at fiducial 
    mass log M = 10.5. We include the compilation fit from Speagle et al. (2014).  
    These comparisons are mainly intended to determine whether the prior for 
    the logSFR_MS parameters are sensible. 
    '''
    # z = 0 
    # fit from SDSS group catalog
    logSFR_sdss = lambda mm: (0.5757 * (mm - 10.5) - 0.13868)

    # Lee et al. (2015)
    logSFR_lee = lambda mm: 1.53 - np.log10(1 + (10**mm/10**(10.10))**-1.26)
    # Noeske et al. (2007) 0.85 < z< 1.10 (by eye)
    logSFR_noeske = lambda mm: (1.580 - 1.064)/(11.229 - 10.029)*(mm - 10.0285) + 1.0637
    # Moustakas et al. (2013) 0.8 < z < 1. (by eye)  
    logSFR_primus = lambda mm: (1.3320 - 1.296)/(10.49 - 9.555) * (mm-9.555) + 1.297

    # Speagle's SFMS bestfit
    logSFR_speagle = lambda mm, tt: (0.84 - 0.026 * tt) * mm - (6.51 - 0.11 * tt) 
   
    logSFR_min = lambda mm, zz: logSFR_sdss(mm) + (zz - 0.05) 
    logSFR_max = lambda mm, zz: logSFR_sdss(mm) + 2. * (zz - 0.05)
 
    zarr = np.linspace(0., 1., 5) 
    tarr = UT.t_from_z(zarr) 

    fig = plt.figure(1)
    sub = fig.add_subplot(111) 
    sub.scatter([0.05], [logSFR_sdss(10.5)], label='SDSS group catalog')  
    sub.scatter([0.95], [logSFR_lee(10.5)], label='Lee et al.(2015)')
    sub.scatter([1.00], [logSFR_noeske(10.5)], label='Noeske et al.(2007)')
    sub.scatter([0.97], [logSFR_primus(10.5)], label='Moustakas et al.(2013)')
    sub.plot(zarr, [logSFR_speagle(10.5, tt) for tt in tarr], 
            c='k', lw=2, label='Speagle')  
    sub.plot(zarr, 
            [logSFR_speagle(10.5, tt) - \
                    (logSFR_speagle(10.5, UT.t_from_z(0.05)) - logSFR_sdss(10.5)) 
                    for tt in tarr], 
            c='k', lw=2, ls='--', label='Speagle offset')  

    sub.fill_between(zarr, 
            [logSFR_min(10.5, zz) for zz in zarr], 
            [logSFR_max(10.5, zz) for zz in zarr], label='Prior', alpha=0.5)

    sub.legend(loc='lower right') 
    sub.set_xlim([0., 1.])
    sub.set_xlabel('Redshift ($z$)', fontsize=20) 
    sub.set_ylim([-2, 2])
    sub.set_ylabel('log SFR ($M_{*,fid} = 10^{10.5} M_\odot$)', fontsize=20) 

    fig.savefig(UT.fig_dir()+'test.sfms.Mfid.z.png', bbox_inches='tight')
    plt.close() 
    return None


def GroupCat_fSFMS(): 
    ''' 
    '''
    
    real = Cat.Observations('group_catalog', Mrcut=18, position='central')
    catalog = real.Read()
    # calculate the quenched fraction 
    qf = Obvs.Fq()
    qfrac = qf.Calculate(mass=catalog['mass'], sfr=catalog['sfr'], z=0.05, theta_SFMS={'name': 'linear', 'zslope': 1.05, 'mslope':0.53})
    
    # get SFMS frac fit from LetsTalkAboutQuench 
    fSFMS = fstarforms() 
    _ = fSFMS.fit(catalog['mass'], catalog['sfr'], method='gaussfit')
    fit_logm, fit_fsfms = fSFMS.frac_SFMS()

    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    sub.plot(qfrac[0], qfrac[1], label='quiescent fraction')
    sub.plot(fit_logm, 1.-fit_fsfms, label='1-$f_\mathrm{SFMS}$') 
    sub.set_xlim([9.5, 12.])
    sub.set_xlabel('log $M_*\;\;[M_\odot]$', fontsize=20)
    sub.set_ylim([0., 1.])
    sub.set_ylabel('$f_\mathrm{Q}$', fontsize=20)
    plt.show()
    plt.close()
    return None


if __name__=='__main__': 
    #SFMS_Mfid_z()
    SFMS_comparison()
    #SFMS_z1()
    #GroupCat_fSFMS()
