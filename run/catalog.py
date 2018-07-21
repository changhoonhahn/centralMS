'''
'''
import os
import h5py
import numpy as np
# -- centralms -- 
from centralms import util as UT
from centralms import catalog as Cat
# -- treepm -- 
try: 
    from treepm import subhalo_io 
    from utilities import utility as wetzel_util
except ImportError:
    pass


def buildSubhalos(nsnap0=20, sigma_smhm=0.2, smf_source='li-march', silent=True): 
    '''
    '''
    shcat = Cat.Subhalos(sigma_smhm=sigma_smhm, smf_source=smf_source, nsnap0=nsnap0)
    shcat._Build(silent=silent)
    return None 


def buildCentralSubhalos(nsnap0=20, sigma_smhm=0.2, smf_source='li-march', silent=True): 
    '''
    '''
    shcat = Cat.CentralSubhalos(sigma_smhm=sigma_smhm, smf_source=smf_source, nsnap0=nsnap0)
    shcat._Build(silent=silent)
    return None 


def buildCentralSubhalosDown(nsnap0=20, sigma_smhm=0.2, smf_source='li-march', silent=True): 
    '''
    '''
    shcat = Cat.CentralSubhalos(sigma_smhm=sigma_smhm, smf_source=smf_source, nsnap0=nsnap0)
    shcat.Downsample(silent=silent)
    return None 


def buildSDSSgroupcat(): 
    '''
    '''
    for Mr in [18, 19, 20]: 
        for cs in ['central', 'satellite', 'all']: 
            cat = Cat.SDSSgroupcat(Mrcut=Mr, censat=cs)
            cat._Build(silent=False)
    return None 


if __name__=="__main__":
    #buildSubhalos(nsnap0=15, sigma_smhm=0.2, smf_source='li-march', silent=False)
    buildCentralSubhalos(nsnap0=15, sigma_smhm=0.2, smf_source='li-march', silent=False)
    buildCentralSubhalosDown(nsnap0=15, sigma_smhm=0.2, smf_source='li-march', silent=False)
    #buildSDSSgroupcat()
