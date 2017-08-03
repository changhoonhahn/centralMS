'''
'''
import numpy as np

# -- local --
import catalog as Cat
import evolver as Evol


def model(run, args, **kwargs): 
    ''' model given the ABC run 
    '''
    theta = {}

    if run == 'test0': 
        # args = SFMS_zslope, SFMS_mslope

        # these values were set by cenque project's output
        theta['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
        theta['fq'] = {'name': 'cosmos_tinker'}
        theta['fpq'] = {'slope': -2.079703, 'offset': 1.6153725, 'fidmass': 10.5}
        
        # for simple test 
        theta['mass'] = {'solver': 'euler', 'f_retain': 0.6, 't_step': 0.05} 
        theta['sfh'] = {'name': 'constant_offset'}
        theta['sfh']['nsnap0'] =  kwargs['nsnap0'] 
            
        # SFMS slopes can change 
        theta['sfms'] = {'name': 'linear', 'zslope': args[0], 'mslope': args[1]}

        # load in Subhalo Catalog (pure centrals)
        subhist = Cat.PureCentralHistory(nsnap_ancestor=kwargs['nsnap0'])
        subcat = subhist.Read(downsampled=kwargs['downsampled']) # full sample

        eev = Evol.Evolver(subcat, theta, nsnap0=kwargs['nsnap0'])
        eev.Initiate()
        eev.Evolve() 
    else: 
        raise NotImplementedError

    return eev.SH_catalog
