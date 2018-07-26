import time 
import numpy as np 
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from . import util as UT 
from . import sfh as SFH


def Evolve(shcat, theta): 
    '''
    '''
    # meta data 
    nsnap0 = shcat['metadata']['nsnap0']
    ngal = len(shcat['m.sham'])

    shcat = initSF(shcat, theta) # get SF halos  
    isSF = np.arange(ngal)[shcat['galtype'] == 'sf']

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
    shcat['m.star'][shcat['galtype'] != 'sf'] = shcat['m.sham'][shcat['galtype'] != 'sf']
    
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
    f_sfs = np.clip(f_sfs, 0., 1.) 
    rand = np.random.uniform(0., 1., ngal) 
    isSF = (rand < f_sfs)
    
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
