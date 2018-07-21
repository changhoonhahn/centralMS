import time 
import numpy as np 
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from . import util as UT 
from . import sfh as SFH


def Evolve(shcat, theta): 
    '''
    '''
    shcat = initSF(shcat, theta) # get SF halos  

    # meta data 
    nsnap0 = shcat['metadata']['nsnap0']
    ngal = len(shcat['m.sham'])
    isSF = np.arange(ngal)[shcat['galtype'] == 'SF']

    # initiate logSFR(logM, z) function and keywords
    logSFR_logM_z, sfr_kwargs = SFH.logSFR_initiate(shcat, isSF, 
            theta_sfh=theta['sfh'], theta_sfms=theta['sfms'])

    # get integrated stellar masses 
    logM_integ, logSFRs = _MassSFR_Wrapper(
            shcat, 
            nsnap0, 
            1,  
            isSF=isSF, 
            logSFR_logM_z=logSFR_logM_z, 
            sfr_kwargs=sfr_kwargs,
            theta_sfh=theta['sfh'], 
            theta_sfms=theta['sfms'], 
            theta_mass=theta['mass'])

    shcat['m.star'] = logM_integ[:,-1] # nsnap = 1 
    shcat['sfr'] = logSFRs
    for ii, n_snap in enumerate(range(2, nsnap0)[::-1]): 
        isSF_i = np.where(shcat['nsnap_start'][isSF] == n_snap)[0] 

        shcat['m.star.snap'+str(n_snap)] = logM_integ[:,ii]
        shcat['sfr.snap'+str(n_snap)] = np.repeat(-999., len(logM_integ[:,ii]))
        # assign M*0 and SFR0 
        shcat['m.star.snap'+str(n_snap)][isSF[isSF_i]] = shcat['m.star0'][isSF[isSF_i]]
        shcat['sfr.snap'+str(n_snap)][isSF[isSF_i]] = shcat['sfr0'][isSF[isSF_i]]
    
    for ii, n_snap in enumerate(range(2, nsnap0)[::-1]): 
        isSF_i = np.where(shcat['nsnap_start'][isSF] >= n_snap)[0] 
        
        sfr_tmp = logSFR_logM_z(
                shcat['m.star.snap'+str(n_snap)][isSF],
                UT.z_nsnap(n_snap), 
                **sfr_kwargs)
        shcat['sfr.snap'+str(n_snap)][isSF[isSF_i]] = sfr_tmp[isSF_i]
    # not star-forming nsnap_f M* is just their SHAM M* 
    shcat['m.star'][~isSF] = shcat['m.sham'][~isSF]
    
    #if forTests: 
    #    self.dlogSFR_amp = sfr_kwargs['dlogSFR_amp']
    #    self.tsteps = sfr_kwargs['tsteps']
    return shcat 


def initSF(shcat, theta): 
    '''
    Initialize the "star forming" subhalos. Select 
    "star-forming" subhalos at z~0 using input f_SFMS(M_SHAM). 
     Assumptions: 
    - f_SFMS does not depend on other subhalo properties. 
    - SF galaxies at z~0 have remained on the SFMS since z > 1 
    '''
    # meta data 
    nsnap0 = shcat['metadata']['nsnap0']
    ngal = len(shcat['m.sham'])
        
    # pick SF subhalos based on f_SFS(M_SHAM) at snapshot 1 
    f_sfs = Fsfms(shcat['m.sham'])
    rand = np.random.uniform(0., 1., ngal) 
    isSF = (rand < f_sfs)
    
    # get m.sham at the initial snapshots of the halo 
    shcat['m.star0'] = np.zeros(ngal) # initial SHAM stellar mass 
    shcat['halo.m0'] = np.zeros(ngal) # initial halo mass 
    for i in range(1, nsnap0+1): 
        istart = (shcat['nsnap_start'] == i) # subhalos that being at snapshot i  
        str_snap = ''
        if i != 1: str_snap = '.snap'+str(i) 
        shcat['m.star0'][istart] = shcat['m.sham'+str_snap][istart]
        shcat['halo.m0'][istart] = shcat['halo.m'+str_snap][istart]

    shcat['sfr0'] = np.repeat(-999., ngal) # assign initial SFRs
    dsfr0 = theta['sfms']['sigma'] * np.random.randn(np.sum(isSF))
    shcat['sfr0'][isSF] = SFH.SFR_sfms(
            shcat['m.star0'][isSF],                     # Mstar
            UT.z_nsnap(shcat['nsnap_start'][isSF]),     # redshift 
            theta['sfms']                               # theta of SFMS 
            ) + dsfr0 
    shcat['galtype'] = UT.replicate('', ngal)
    shcat['galtype'][isSF] = 'sf'
    return shcat


def _MassSFR_Wrapper(SHcat, nsnap0, nsnapf, isSF=None, logSFR_logM_z=None, sfr_kwargs=None, **theta): 
    ''' Evolve galaxies that remain star-forming throughout the snapshots. 
    '''
    # parse theta 
    theta_mass = theta['theta_mass']
    theta_sfh = theta['theta_sfh']
    theta_sfms = theta['theta_sfms']

    # precompute z(t_cosmic) 
    z_table, t_table = UT.zt_table()     
    #z_of_t = interp1d(t_table, z_table, kind='cubic') 
    z_of_t = lambda tt: UT.z_of_t(tt, deg=6)
    
    #t_s = time.time()     
    #dlogmdt_kwargs['dSFR'] = dlogmdt_kwargs['dSFR'][0]

    # now solve M*, SFR ODE 
    dlogmdt_kwargs = {}
    dlogmdt_kwargs['logsfr_M_z'] = logSFR_logM_z 
    dlogmdt_kwargs['f_retain'] = theta_mass['f_retain']
    dlogmdt_kwargs['zoft'] = z_of_t
    
    # choose ODE solver
    if theta_mass['solver'] == 'euler': # Forward euler
        f_ode = SFH.ODE_Euler
    elif theta_mass['solver'] == 'scipy':  # scipy ODE solver
        f_ode = odeint
    else: 
        #theta_mass['solver'] == 'rk4':     # RK4 (does not work for stiff ODEs -- thanks geoff!)
        raise ValueError

    logM_integ = np.tile(-999., (len(SHcat['galtype']), nsnap0 - nsnapf))
    
    dlogmdt_kwarg_list = []
    for nn in range(nsnapf+1, nsnap0+1)[::-1]: 
        # starts at n_snap = nn 
        isStart = np.where(SHcat['nsnap_start'][isSF] == nn)  
    
        if theta_mass['solver'] != 'scipy':  
            dlogmdt_kwarg = dlogmdt_kwargs.copy()
            
            for k in sfr_kwargs.keys(): 
                if isinstance(sfr_kwargs[k], np.ndarray): 
                    dlogmdt_kwarg[k] = sfr_kwargs[k][isStart]
                else: 
                    dlogmdt_kwarg[k] = sfr_kwargs[k]

            dlogmdt_kwarg_list.append(dlogmdt_kwarg)
            del dlogmdt_kwarg
        else:
            sfr_kwarg = {}
            for k in sfr_kwargs.keys(): 
                if isinstance(sfr_kwargs[k], np.ndarray): 
                    sfr_kwarg[k] = sfr_kwargs[k][isStart]
                else: 
                    sfr_kwarg[k] = sfr_kwargs[k]
            
            dlogmdt_arg = (
                    dlogmdt_kwargs['logsfr_M_z'],
                    dlogmdt_kwargs['f_retain'],
                    dlogmdt_kwargs['zoft'],
                    sfr_kwarg
                    )
            dlogmdt_kwarg_list.append(dlogmdt_arg)
            del dlogmdt_arg

    #t_s = time.time() 
    for i_n, nn in enumerate(range(nsnapf+1, nsnap0+1)[::-1]): 
        # starts at n_snap = nn 
        isStart = np.where(SHcat['nsnap_start'][isSF] == nn)  
        
        if theta_mass['solver'] != 'scipy': 
            tmp_logM_integ = f_ode(
                    SFH.dlogMdt,                            # dy/dt
                    SHcat['m.star0'][isSF[isStart]],        # logM0
                    t_table[nsnapf:nn+1][::-1],             # t_final 
                    theta_mass['t_step'],                   # time step
                    **dlogmdt_kwarg_list[i_n]) 
        else: 
            print '=================================='
            print '===========SCIPY ODEINT==========='
            tmp_logM_integ = f_ode(
                    SFH.dlogMdt_scipy,                      # dy/dt
                    SHcat['m.star0'][isSF[isStart]],        # logM0
                    t_table[nsnapf:nn+1][::-1],             # t_final 
                    args=dlogmdt_kwarg_list[i_n]) 

        logM_integ[isSF[isStart], nsnap0-nn:] = tmp_logM_integ.T[:,1:]

    isStart = np.where(SHcat['nsnap_start'][isSF] == 1)  
    logM_integ[isSF[isStart], -1] = SHcat['m.star0'][isSF[isStart]]
    #print time.time() - t_s
    
    # log(SFR) @ nsnapf
    logSFRs = np.repeat(-999., len(SHcat['galtype']))
    logSFRs[isSF] = logSFR_logM_z(logM_integ[isSF, -1], UT.z_nsnap(nsnapf), **sfr_kwargs) 
    
    return logM_integ, logSFRs


def defaultTheta(sfh): 
    ''' Return generic default parameter values
    '''
    theta = {} 

    theta['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
    theta['sfms'] = {'name': 'flex', 'zslope': 1.05, 'mslope':0.58, 'offset': -0.1, 'sigma': 0.3}
    theta['fq'] = {'name': 'cosmos_tinker'}
    theta['fpq'] = {'slope': -2.079703, 'offset': 1.6153725, 'fidmass': 10.5}
    theta['mass'] = {'solver': 'euler', 'f_retain': 0.6, 't_step': 0.05} 
    
    theta['sfh'] = {'name': sfh}
    if sfh == 'constant_offset': 
        theta['sfh']['nsnap0'] = 15 
    elif sfh == 'corr_constant_offset':
        theta['sfh']['m.kind'] = 'm.star'
        theta['sfh']['dm.kind'] = 0.01 
        theta['sfh']['sig_abias'] = 0.3 
    elif sfh == 'random_step': 
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['sigma'] = 0.3 
    elif sfh == 'random_step_fluct': 
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['sigma'] = 0.3 
    elif sfh == 'random_step_abias': 
        theta['sfh']['dt_min'] = 0.25 
        theta['sfh']['dt_max'] = 0.25 
        theta['sfh']['sigma_tot'] = 0.3 
        theta['sfh']['sigma_corr'] = 0.29
    elif sfh == 'random_step_abias2': 
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['t_abias'] = 2. # Gyr
        theta['sfh']['sigma_tot'] = 0.3 
        theta['sfh']['sigma_corr'] = 0.29
    elif sfh == 'random_step_abias_delay': 
        theta['sfh']['dt_min'] = 0.5 
        theta['sfh']['dt_max'] = 0.5 
        theta['sfh']['sigma_tot'] = 0.3 
        theta['sfh']['sigma_corr'] = 0.2
        theta['sfh']['dt_delay'] = 1. # Gyr 
        theta['sfh']['dz_dMh'] = 0.5 
    else: 
        raise NotImplementedError

    return theta 


def Fsfms(mm): 
    ''' Star Formation Main Sequence fraction as a function of log M*.
    See paper.py to see how f_SFMS was estimated for each stellar mass 
    from the SDSS Group Catalog. 
    '''
    return -0.634 * mm + 6.898


"""
    class Evolver(object): 
        def __init__(self, PCH_catalog, theta, nsnap0=20): 
            ''' class object that takes a given catalog of galaxy/subhalo snapshots 
            then constructs star formation histories for them. 

            Parameters
            ----------
            PCH_catalog : (obj)
                Object with catalog of subhalo accretion histories
        
            theta : (dictionary) 
                Dictionary that specifies the inital conditions and evolution
                of galaxy properties

            nsnap0 : (int) 
                The oldest snapshot. Default is nsnap = 20, which corresponds
                to z = 1.661 .
            '''
            self.nsnap0 = nsnap0
            self._UnpackTheta(theta) # unpack the parameters to be usable
        
            # store subhalo catalog object described in catalog.py
            self.SH_catalog = PCH_catalog

        def newEvolve(self, forTests=False): 
            ''' Evolve the galaxies from initial conditions specified in self.InitSF()
            '''
            # SF galaxies selected by self.InitSF
            isSF = np.where(self.SH_catalog['gclass'] == 'sf')[0] 
        
            # initiate logSFR(logM, z) function and keywords
            logSFR_logM_z, sfr_kwargs = SFH.logSFR_initiate(self.SH_catalog, isSF, 
                    theta_sfh=self.theta_sfh, theta_sfms=self.theta_sfms)
            if forTests: 
                self.dlogSFR_amp = sfr_kwargs['dlogSFR_amp']
                self.tsteps = sfr_kwargs['tsteps']

            # get integrated stellar masses 
            logM_integ, logSFRs = _MassSFR_Wrapper(self.SH_catalog, self.nsnap0, 1,  
                    isSF=isSF, logSFR_logM_z=logSFR_logM_z, sfr_kwargs=sfr_kwargs,
                    theta_sfh=self.theta_sfh, theta_sfms=self.theta_sfms, theta_mass=self.theta_mass)

            # save into SH catalog
            self.SH_catalog['m.star'] = logM_integ[:,-1] # nsnap = 1 
            self.SH_catalog['sfr'] = logSFRs

            # assign SFR and Mstar to other snapshots
            for ii, n_snap in enumerate(range(2, self.nsnap0)[::-1]): 
                isSF_i = np.where(self.SH_catalog['nsnap_start'][isSF] == n_snap)[0] 

                self.SH_catalog['m.star.snap'+str(n_snap)] = logM_integ[:,ii]
                self.SH_catalog['sfr.snap'+str(n_snap)] = np.repeat(-999., len(logM_integ[:,ii]))
                # assign M*0 and SFR0 
                self.SH_catalog['m.star.snap'+str(n_snap)][isSF[isSF_i]] = self.SH_catalog['m.star0'][isSF[isSF_i]]
                self.SH_catalog['sfr.snap'+str(n_snap)][isSF[isSF_i]] = self.SH_catalog['sfr0'][isSF[isSF_i]]
            
            for ii, n_snap in enumerate(range(2, self.nsnap0)[::-1]): 
                isSF_i = np.where(self.SH_catalog['nsnap_start'][isSF] >= n_snap)[0] 
                
                sfr_tmp = logSFR_logM_z(self.SH_catalog['m.star.snap'+str(n_snap)][isSF],
                        UT.z_nsnap(n_snap), **sfr_kwargs)
                self.SH_catalog['sfr.snap'+str(n_snap)][isSF[isSF_i]] = sfr_tmp[isSF_i]
            
            # not star-forming nsnap_f M* is just their SHAM M* 
            isNotSF = np.where(self.SH_catalog['gclass'] != 'sf')
            self.SH_catalog['m.star'][isNotSF] = self.SH_catalog['m.sham'][isNotSF]

            # store theta values 
            for k in self.__dict__.keys(): 
                if 'theta_' in k: 
                    self.SH_catalog[k] = getattr(self, k)
            return None

        def InitSF(self): 
            ''' Initialize the "star forming" subhalos. Select 
            "star-forming" subhalos at z~0 using input f_SFMS(M_SHAM). 
            Assumptions: 
            - f_SFMS does not depend on other subhalo properties. 
            - SF galaxies at z~0 have remained on the SFMS since z > 1 
            '''
            self.SH_catalog['nsnap0'] = self.nsnap0

            ngal = len(self.SH_catalog['m.star'])
            
            # pick SF subhalos based on f_SFMS(M_SHAM) at snapshot 1 
            f_sfms = Fsfms(self.SH_catalog['m.star']) 
            rand = np.random.uniform(0., 1., ngal) 
            isSF = np.where(rand < f_sfms)

            # determine initial subhalo and stellar mass
            m0, hm0 = np.zeros(ngal), np.zeros(ngal)
            for i in range(1, self.nsnap0+1): # "m.star" from subhalo catalog is from SHAM
                if i == 1: 
                    sm_tag = 'm.sham'
                    hm_tag = 'halo.m'
                    self.SH_catalog[sm_tag] = self.SH_catalog.pop('m.star')  # store sham masses
                else: 
                    sm_tag = 'm.sham.snap'+str(i)
                    hm_tag = 'halo.m.snap'+str(i)
                    self.SH_catalog[sm_tag] = self.SH_catalog.pop('m.star.snap'+str(i))

                started = np.where(self.SH_catalog['nsnap_start'] == i) # subhalos that being at snapshot i  

                m0[started] = self.SH_catalog[sm_tag][started]
                hm0[started] = self.SH_catalog[hm_tag][started]

            self.SH_catalog['m.star0'] = m0 # initial SHAM stellar mass 
            self.SH_catalog['halo.m0'] = hm0 # initial subhalo mass 
            
            # assign initial SFRs
            self.SH_catalog['sfr0'] = np.repeat(-999., ngal)
            self.SH_catalog['sfr0'][isSF] = SFH.SFR_sfms(m0[isSF], UT.z_nsnap(self.SH_catalog['nsnap_start'][isSF]), self.theta_sfms) + self.theta_sfms['sigma'] * np.random.randn(len(isSF[0]))
            self.SH_catalog['gclass'] = UT.replicate('', ngal)
            self.SH_catalog['gclass'][isSF] = 'sf'
            return None
        
        def _UnpackTheta(self, theta): 
            ''' Unpack the parameters into easier to use pieces. 
            '''
            # green valley parameters (slope and y-int of f_gv(M*)) 
            self.theta_gv = theta['gv']
            # SFMS parameters (slope and y-int of f_gv(M*)) 
            self.theta_sfms = theta['sfms']
            # fq parameters (name of fQ model) 
            self.theta_fq = theta['fq']
            # f_PQ parameters 
            self.theta_fpq = theta['fpq'] 
        
            self.theta_mass = theta['mass']

            self.theta_sfh = theta['sfh']

            return None
"""
