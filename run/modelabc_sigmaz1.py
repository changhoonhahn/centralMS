#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys 
import pickle
import numpy as np 
from mpi4py import MPI
# --- centralms --- 
from centralms import util as UT
from centralms import abcee as ABC
from centralms import observables as Obvs 

run = sys.argv[1]
sigmaz1 = float(sys.argv[2])
T = int(sys.argv[3]) 

nsnap0 = 15
sigma_smhm = sigmaz1 
downsampled = '20'

savekeys = ['m.star', 'halo.m', 'm.max', 'weights', 'sfr', 'galtype']
    
abc_dir = ''.join([UT.dat_dir(), 'abc/', run, '/model/'])  # directory where all the ABC files are stored
if not os.path.exists(abc_dir): # make directory if it doesn't exist 
    try: 
        os.makedirs(abc_dir)
    except OSError:
        pass 

# save the median theta separately (evaluate it a bunch of times) 
abcout = ABC.readABC(run, T) # read in the abc particles 
theta_med = [np.median(abcout['theta'][:,i]) for i in range(abcout['theta'].shape[1])]
for i in range(10):  
    subcat_sim = ABC.model(run, theta_med, nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 

    fname = ''.join([abc_dir, 'model.theta_median', str(i), '.t', str(T), '.p'])
    out = {} 
    for key in savekeys: 
        out[key] = subcat_sim[key]
    pickle.dump(out, open(fname, 'wb'))

COMM = MPI.COMM_WORLD # default communicator

def _split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

def model_thetai(i):
    print('%s iteration %i : model ABC theta %i' % (run, T, i))
    subcat_sim_i = ABC.model(run, abcout['theta'][i], nsnap0=nsnap0, sigma_smhm=sigma_smhm, downsampled=downsampled) 
    fname = ''.join([abc_dir, 'model.theta', str(i), '.t', str(T), '.p'])
    out = {} 
    for key in savekeys: 
        out[key] = subcat_sim[key]
    pickle.dump(out, open(fname, 'wb'))
    return i 

if COMM.rank == 0:
    jobs = range(len(abcout['w']))
    jobs = _split(jobs, COMM.size)
else: 
    jobs = None 

# scatter jobs across cores
jobs = COMM.scatter(jobs, root=0) 

results = [] 
for job in jobs: 
    results.append(model_thetai(job))    

# gather the results
results = MPI.COMM_WORLD.gather(results, root=0) 

if COMM.rank == 0: 
    results = [_i for temp in results for _i in temp]
    print("Results:", results)
