'''
'''
import time 
import numpy as np 
# -- centralms -- 
from centralms import util as UT 
from centralms import catalog as Cat
from centralms import evolver as Evo
from centralms import observables as Obvs
# -- plotting --
import corner as DFM 
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


def initSF(): 
    sh = Cat.CentralSubhalos(
            sigma_smhm=0.2, 
            smf_source='li-march', 
            nsnap0=15) 
    shcat = sh.Read(downsampled='20') 
    tt =  Evo.defaultTheta('constant_offset') 

    t0 = time.time()
    shcat = Evo.initSF(shcat, tt) 
    print('initSF takes %f sec' % (time.time()-t0))

    mbins = np.linspace(9., 12., 30)
    fq = np.zeros(len(mbins)-1)
    for i_m in range(len(mbins)-1): 
        inmbin = (shcat['m.sham'] > mbins[i_m]) & (shcat['m.sham'] <= mbins[i_m+1]) 
        if np.sum(inmbin) > 0:
            fq[i_m] = float(np.sum(shcat['weights'][inmbin & (shcat['galtype'] == 'sf')]))/float(np.sum(shcat['weights'][inmbin]))
        
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.scatter(0.5*(mbins[1:] + mbins[:-1]), fq, c='C0', s=20) 
    sub.plot(mbins, Evo.Fsfms(mbins), c='k', ls='--', lw=2)
    sub.set_xlabel(r'$\log\,M_*$', fontsize=25)
    sub.set_xlim([9., 12.]) 
    sub.set_ylabel(r'$f_\mathrm{SF}$', fontsize=25)
    sub.set_ylim([0., 1.]) 
    fig.savefig(''.join([UT.fig_dir(), 'initsf.png']), bbox_inches='tight') 
    plt.close() 
    return None 


def Evolve(sfh='constant_offset'): 
    sh = Cat.CentralSubhalos(
            sigma_smhm=0.2, 
            smf_source='li-march', 
            nsnap0=15) 
    shcat = sh.Read(downsampled='20') 
    tt =  Evo.defaultTheta(sfh) 

    t0 = time.time()
    shcat = Evo.Evolve(shcat, tt) 
    print('Evolve takes %f sec' % (time.time()-t0))

    nsnap0 = shcat['metadata']['nsnap0'] 
    isSF = (shcat['galtype'] == 'sf')
        
    fig = plt.figure(figsize=(12,4))
    # stellar mass function 
    sub = fig.add_subplot(131)
    for n in range(2, nsnap0)[::-1]: 
        # identify SF population at snapshot
        smf_sf = Obvs.getMF(shcat['m.star.snap'+str(n)][isSF], 
                weights=shcat['weights'][isSF])
        sub.plot(smf_sf[0], smf_sf[1], lw=2, c='b', alpha=0.05 * (21. - n))        

    smf_sf = Obvs.getMF(shcat['m.star'][isSF], weights=shcat['weights'][isSF])
    sub.plot(smf_sf[0], smf_sf[1], lw=3, c='b', ls='-', label='Integrated')

    smf_sf_msham = Obvs.getMF(shcat['m.sham'][isSF], weights=shcat['weights'][isSF])
    sub.plot(smf_sf_msham[0], smf_sf_msham[1], lw=3, c='k', ls='--', label='SHAM')
    sub.set_xlabel('Stellar Masses $(\mathcal{M}_*)$', fontsize=25)
    sub.set_xlim([9., 11.5])
    sub.set_ylim([1e-5, 10**-1.75])
    sub.set_yscale('log')
    sub.set_ylabel('log $\Phi$', fontsize=25)
    sub.legend(loc='upper right', prop={'size': 20}) 
    
    sub = fig.add_subplot(132) # Star Forming Sequence 
    DFM.hist2d(
            shcat['m.star'][isSF], 
            shcat['sfr'][isSF], 
            weights=shcat['weights'][isSF], 
            levels=[0.68, 0.95], range=[[9., 12.], [-3., 1.]], 
            bins=20, plot_datapoints=False, fill_contours=False, plot_density=True, ax=sub) 
    sub.set_xlabel('log $(\; M_*\; [M_\odot]\;)$', fontsize=25)
    sub.set_xlim([9., 11.5])
    sub.set_xticks([9., 10., 11.]) 
    sub.set_ylabel('log $(\;\mathrm{SFR}\;[M_\odot/\mathrm{yr}])$', fontsize=25)
    sub.set_ylim([-2.5, 1.])
    sub.set_yticks([-2., -1., 0., 1.])

    # stellar to halo mass relation 
    sub = fig.add_subplot(133)
    smhmr = Obvs.Smhmr()
    m_mid, mu_mstar, sig_mstar, cnts = smhmr.Calculate(shcat['halo.m'][isSF], shcat['m.star'][isSF])
    sub.errorbar(m_mid, mu_mstar, yerr=sig_mstar)
    sub.fill_between(m_mid, mu_mstar - 0.2, mu_mstar + 0.2, color='k', alpha=0.25, linewidth=0, edgecolor=None)

    sub.set_xlabel('Halo Mass $(\mathcal{M}_{halo})$', fontsize=25)
    sub.set_xlim([11., 14.])
    sub.set_ylabel('Stellar Mass $(\mathcal{M}_*)$', fontsize=25)
    sub.set_ylim([9., 12.])

    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig(''.join([UT.fig_dir(), 'evolver.', sfh, '.png']), bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__": 
    Evolve()
    Evolve('random_step')
