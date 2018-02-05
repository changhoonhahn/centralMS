'''
wrapper for evaluating Model(ABC particle)
'''
import env 
import sys 
from abcee import model_ABCparticle 

#restart = int(sys.argv[1])
#if restart == 0: 

# ABC run name 
abcrun = sys.argv[1]
print 'Run ', abcrun

# iterations number
Niter = int(sys.argv[2])
print 'N_iterations = ', Niter

# plus some hardcoded kwargs
nsnap0=15 # starting snapshot 
sigma_smhm=0.2

model_ABCparticle(abcrun, Niter, nsnap0=nsnap0, sigma_smhm=sigma_smhm) 
