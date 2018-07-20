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


if __name__=="__main__":
    buildSubhalos(nsnap0=15, sigma_smhm=0.2, smf_source='li-march', silent=False)
