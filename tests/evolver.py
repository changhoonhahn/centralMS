'''
'''
import time 
import numpy as np 
# -- centralms -- 
from centralms import util as UT 
from centralms import catalog as Cat
from centralms import evolver as Evo
# -- plotting --
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


def Evolve(): 
    sh = Cat.CentralSubhalos(
            sigma_smhm=0.2, 
            smf_source='li-march', 
            nsnap0=15) 
    shcat = sh.Read(downsampled='20') 
    tt =  Evo.defaultTheta('constant_offset') 

    t0 = time.time()
    shcat = Evo.Evolve(shcat, tt) 
    print('Evolve takes %f sec' % (time.time()-t0))
    return None 


if __name__=="__main__": 
    Evolve()
