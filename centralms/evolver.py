'''



'''
import time 
import numpy as np 
from scipy.interpolate import interp1d
from scipy.integrate import odeint

import util as UT 
import sfh as SFH
import observables as Obvs


def defaultTheta(sfh): 
    ''' Return generic default parameter values
    '''
    theta = {} 

    theta['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
    theta['sfms'] = {'name': 'linear', 
            'zslope': 1.05,#14, 
            'mslope':0.53}
    #theta['sfms'] = {'name': 'kinked', 'zslope': 1.1, 'mslope_high':0.53, 'mslope_low': 0.65}
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
    else: 
        raise NotImplementedError

    return theta 


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

    def Evolve(self, forTests=False): 
        ''' Evolve the galaxies from initial conditions specified in self.Initiate()
        '''
        # galaxies in the subhalo snapshots (SHcat) that are SF throughout 
        isSF = np.where(
                (self.SH_catalog['gclass'] == 'star-forming') & 
                (self.SH_catalog['weights'] > 0.))[0] # only includes galaxies with w > 0 
    
        # initiate logSFR(logM, z) function and keywords
        logSFR_logM_z, sfr_kwargs = SFH.logSFR_initiate(self.SH_catalog, isSF, 
                theta_sfh=self.theta_sfh, theta_sfms=self.theta_sfms)
        if forTests: 
            self.sfr_kwargs = sfr_kwargs
            #self.logSFR_logM_z = logSFR_logM_z

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

            self.SH_catalog['snapshot'+str(n_snap)+'_m.star'] = logM_integ[:,ii]

            self.SH_catalog['snapshot'+str(n_snap)+'_sfr'] = \
                    np.repeat(-999., len(logM_integ[:,ii]))
            # assign M*0 and SFR0 
            self.SH_catalog['snapshot'+str(n_snap)+'_m.star'][isSF[isSF_i]] = \
                    self.SH_catalog['m.star0'][isSF[isSF_i]]
            self.SH_catalog['snapshot'+str(n_snap)+'_sfr'][isSF[isSF_i]] = \
                    self.SH_catalog['sfr0'][isSF[isSF_i]]
        
        for ii, n_snap in enumerate(range(2, self.nsnap0)[::-1]): 
            isSF_i = np.where(self.SH_catalog['nsnap_start'][isSF] >= n_snap)[0] 
            
            sfr_tmp = logSFR_logM_z(self.SH_catalog['snapshot'+str(n_snap)+'_m.star'][isSF],
                    UT.z_nsnap(n_snap), **sfr_kwargs)

            self.SH_catalog['snapshot'+str(n_snap)+'_sfr'][isSF[isSF_i]] = sfr_tmp[isSF_i]

        return None

    def Initiate(self): 
        ''' Assign initial conditions to galaxies at their nsnap_start. More 
        specifically assign SFRs to galaxies at snpashot self.nsnap0 based 
        on their SHAM stellar masses, theta_gv, theta_sfms, and theta_fq. 

        Details 
        -------
        * Assign SFRs to galaxies *with* weights
        * Although most nsnap_start corresponds to nsnap0. They do not 
            necessary have to correspond. 
        '''
        self.SH_catalog['nsnap0'] = self.nsnap0

        ngal = len(self.SH_catalog['m.star'])
        
        # first determine initial subhalo and stellar mass
        m0, hm0 = np.zeros(ngal), np.zeros(ngal)
        for i in range(1, self.nsnap0+1): # "m.star" from subhalo catalog is from SHAM
            if i == 1: 
                sm_tag = 'm.sham'
                hm_tag = 'halo.m'
                self.SH_catalog[sm_tag] = self.SH_catalog.pop('m.star')  # store sham masses
            else: 
                sm_tag = 'snapshot'+str(i)+'_m.sham'
                hm_tag = 'snapshot'+str(i)+'_halo.m'
                self.SH_catalog[sm_tag] = self.SH_catalog.pop('snapshot'+str(i)+'_m.star') 

            started = np.where(self.SH_catalog['nsnap_start'] == i) # subhalos that being at snapshot i  

            m0[started] = self.SH_catalog[sm_tag][started]
            hm0[started] = self.SH_catalog[hm_tag][started]

        self.SH_catalog['m.star0'] = m0 # initial SHAM stellar mass 
        self.SH_catalog['halo.m0'] = hm0 # initial subhalo mass 

        keep = np.where(self.SH_catalog['weights'] > 0) # only galaxies that are weighted
    
        t_s = time.time()
        # assign SFRs at z_star
        sfr_out = assignSFRs(
                m0[keep], 
                UT.z_nsnap(self.SH_catalog['nsnap_start'][keep]), 
                self.SH_catalog['weights'][keep],
                theta_GV = self.theta_gv, 
                theta_SFMS = self.theta_sfms,
                theta_FQ = self.theta_fq) 

        # save z0 SFR into self.SH_catalog 
        for key in sfr_out.keys(): 
            if key == 'SFR': 
                k = 'sfr0'
            elif key == 'Gclass': 
                k = 'gclass0' 
            else: 
                k = key 
            self.SH_catalog[k] = UT.replicate(sfr_out[key], len(self.SH_catalog['m.sham']))
            self.SH_catalog[k][keep] = sfr_out.pop(key)
        #for key in sfr_out: 
        #    self.SH_catalog['snapshot'+str(self.nsnap0)+'_'+key.lower()] = \
        #            UT.replicate(sfr_out[key], len(self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.sham']))
        #    self.SH_catalog['snapshot'+str(self.nsnap0)+'_'+key.lower()] = sfr_out[key][keep] 
        self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.star'] = self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.sham'] 
        
        # Propagate P_Q from z0 to zf and pick out quenching galaxies
        gclass, nsnap_quench = _pickSF(self.SH_catalog, nsnap0=self.nsnap0, theta_fq=self.theta_fq, theta_fpq=self.theta_fpq)

        self.SH_catalog['gclass'] = gclass
        self.SH_catalog['nsnap_quench'] = nsnap_quench
        
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
    
    t_s = time.time()     
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

    logM_integ = np.tile(-999., (len(SHcat['gclass']), nsnap0 - nsnapf))
    
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

    t_s = time.time() 
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
    print time.time() - t_s
    
    # log(SFR) @ nsnapf
    logSFRs = np.repeat(-999., len(SHcat['gclass']))
    logSFRs[isSF] = logSFR_logM_z(logM_integ[isSF, -1], UT.z_nsnap(nsnapf), **sfr_kwargs) 

    return logM_integ, logSFRs


def _pickSF(SHcat, nsnap0=20, theta_fq=None, theta_fpq=None): 
    ''' Take subhalo catalog and then based on P_Q(M_sham, z) determine, which
    galaxies quench or stay star-forming
    ''' 
    nsnap_quench = np.repeat(-999, len(SHcat['weights']))  # snapshot where galaxy quenches 
    gclass = SHcat['gclass0']
    
    # identify SF galaxies @ nsnap0
    isSF = np.where((gclass == 'star-forming') & (SHcat['weights'] > 0.))
    
    qf = Obvs.Fq() # qf object

    #  go from nsnap_0 --> nsnap_f and identify the quenching galaxies based on P_Q
    for n in range(2, nsnap0+1)[::-1]: 

        z_i = UT.z_nsnap(n) # snapshot redshift

        m_sham = SHcat['snapshot'+str(n)+'_m.sham'][isSF] # M_sham of SF galaxies

        t_step = UT.t_nsnap(n - 1) - UT.t_nsnap(n) # Gyr between Snapshot n and n-1 
        
        # quenching probabily 
        # P_Q^cen = f_PQ * ( d(n_Q)/dt 1/n_SF ) 
        mf = _SnapCat_mf(SHcat, n, prop='m.sham')     # MF 
        dmf_dt = _SnapCat_dmfdt(SHcat, n, prop='m.sham')  # dMF/dt
        assert np.array_equal(mf[0], dmf_dt[0])
    
        # (P_Q fiducial) * (1-fq) 
        Pq_M_fid = (qf.dfQ_dz(mf[0], z_i, lit=theta_fq['name']) / UT.dt_dz(z_i) +
                qf.model(mf[0], z_i, lit=theta_fq['name']) * dmf_dt[1] / mf[1])

        Pq_M_fid_interp = interp1d(mf[0], Pq_M_fid, fill_value='extrapolate') # interpolate

        Pq_M = lambda mm: t_step * _f_PQ(mm, theta_fpq['slope'], theta_fpq['fidmass'], theta_fpq['offset']) *  \
                Pq_M_fid_interp(mm) / (1. - qf.model(mm, z_i, lit=theta_fq['name']))

        hasmass = np.where(m_sham > 0.) 
        Pq_Msham = Pq_M(m_sham[hasmass])
        rand_Pq = np.random.uniform(0., 1., len(hasmass[0])) 
        
        # galaxies that quench between snpashots n and n-1 
        quenches = np.where(rand_Pq < Pq_Msham)  # these SFing galaxies quench
        gclass[isSF[0][hasmass[0][quenches]]] = 'quenching'
        nsnap_quench[isSF[0][hasmass[0][quenches]]] = n
        
        isSF = np.where((gclass == 'star-forming') & (SHcat['weights'] > 0.)) # update is SF

    return [gclass, nsnap_quench]


def assignSFRs(masses, zs, ws, theta_GV=None, theta_SFMS=None, theta_FQ=None): 
    ''' Given stellar masses, zs, and parameters that describe the 
    green valley, SFMS, and FQ return SFRs. The SFRs are assigned without
    consideration of weights. 

    Details: 
    -------
    - Designates a fraction of galaxies as green valley "quenching" galaxies based on theta_GV
    
    Parameters
    ----------
    masses : (array)
        Array that of stellar masses
    
    zs : (array)
        Array that of stellar zs 

    theta_XX : (dict) 
        Dictary that specifies XX property  

    Return: 
    ------
    output : (dict) 
        Dictionary that specifies the following 
    '''
    np.random.seed()
    qf = Obvs.Fq()   # initialize quiescent fraction class 

    # check inputs 
    if theta_GV is None: 
        raise ValueError("Specify green valley parameters")
    if theta_SFMS is None: 
        raise ValueError("Specify Star-Forming Main Sequence parameters")
    if theta_FQ is None: 
        raise ValueError("Specify Quiescent Fraction parameters")

    assert len(masses) > 0  
    assert len(masses) == len(zs) 
    assert len(masses) == len(ws) 

    ngal = len(masses)   # N_gals

    # set up output 
    output = {} 
    for key in ['SFR', 'Gclass', 'MQ']: 
        if key != 'Gclass':  
            output[key] = UT.replicate(-999., ngal)
        else: 
            output[key] = UT.replicate('', ngal)
    
    # Assign Green valley galaxies 
    f_gv = lambda mm: theta_GV['slope'] * (mm - theta_GV['fidmass']) + theta_GV['offset'] # f_GV 
    
    rand = np.random.uniform(0., 1., ngal)
    isgreen = np.where(rand < f_gv(masses))
    output['Gclass'][isgreen] = 'quenching'
    output['MQ'][isgreen] = masses[isgreen]         # M* at quenching
    output['SFR'][isgreen] = np.random.uniform(     # sample SSFR from uniform distribution 
            Obvs.SSFR_Qpeak(masses[isgreen]),       # between SSFR_Qpeak
            Obvs.SSFR_SFMS(masses[isgreen], zs[isgreen], theta_SFMS=theta_SFMS),                
            len(isgreen[0])) + masses[isgreen]      # and SSFR_SFMS 

    notgreen = rand >= f_gv(masses)
    isnotgreen = np.where(notgreen)[0]
    fQ_true = np.zeros(ngal)

    for zz in np.unique(zs): 
        isgreen_z = np.where((rand < f_gv(masses)) & (zs == zz))

        # GV galaxy queiscent fraction 
        gv_fQ = qf.Calculate(mass=masses[isgreen_z], sfr=output['SFR'][isgreen_z], z=zz, 
                mass_bins=np.arange(masses.min()-0.1, masses.max()+0.2, 0.2), theta_SFMS=theta_SFMS)
    
        # quiescent galaxies 
        fQ_gv = interp1d(gv_fQ[0], gv_fQ[1] * f_gv(gv_fQ[0]))

        isnotgreen_z = np.where(notgreen & (zs == zz))[0]
        fQ_true[isnotgreen_z] = qf.model(masses[isnotgreen_z], zz, lit=theta_FQ['name']) - fQ_gv(masses[isnotgreen_z])
    
    rand2 = np.random.uniform(0., 1., len(isnotgreen))

    isq = isnotgreen[np.where(rand2 < fQ_true[isnotgreen])]
    Nq = len(isq)

    output['Gclass'][isq] = 'quiescent'
    output['SFR'][isq] = Obvs.SSFR_Qpeak(masses[isq]) + \
            np.random.randn(Nq) * Obvs.sigSSFR_Qpeak(masses[isq]) + masses[isq]
    
    # star-forming galaxies 
    issf = np.where(output['Gclass'] == '')
    Nsf = len(issf[0])

    output['Gclass'][issf] = 'star-forming'
    output['SFR'][issf] = Obvs.SSFR_SFMS(masses[issf], zs[issf], theta_SFMS=theta_SFMS) + \
            np.random.randn(Nsf) * Obvs.sigSSFR_SFMS(masses[issf]) + \
            masses[issf]
    
    return output


def _f_PQ(mm, slope, fidmass, offset): 
    fpq = slope * (mm - fidmass) + offset
    fpq.clip(min=1.)
    return fpq


def _SnapCat_mf(snapcat, nsnap, prop='m.sham'):
    # Calculate mass function (mass specified by prop) 
    kk = 'snapshot'+str(nsnap)+'_'+prop
    if kk not in snapcat.keys() or 'weights' not in snapcat.keys():
        raise ValueError
    mf = Obvs.getMF(snapcat[kk], weights=snapcat['weights'])
    return mf 


def _SnapCat_dmfdt(snapcat, nsnap, prop='m.sham'): 
    ''' Calculate the derivative of the mass function (mass specified by prop) 
    '''
    k1 = 'snapshot'+str(nsnap-1)+'_'+prop      # check keys
    k2 = 'snapshot'+str(nsnap+1)+'_'+prop
    if k1 not in snapcat.keys() and k2 not in snapcat.keys():
        raise ValueError
    elif k1 not in snapcat.keys(): 
        k1 = 'snapshot'+str(nsnap)+'_'+prop      # check keys
        dt = UT.t_nsnap(nsnap) - UT.t_nsnap(nsnap-1)
    elif k2 not in snapcat.keys(): 
        k2 = 'snapshot'+str(nsnap)+'_'+prop      # check keys
        dt = UT.t_nsnap(nsnap+1) - UT.t_nsnap(nsnap)
    else: 
        dt = UT.t_nsnap(nsnap+1) - UT.t_nsnap(nsnap-1)

    if 'weights' not in snapcat.keys():
        raise ValueError

    mf1 = Obvs.getMF(snapcat[k1], weights=snapcat['weights'])
    mf2 = Obvs.getMF(snapcat[k2], weights=snapcat['weights'])
    assert np.array_equal(mf1[0], mf2[0])

    return [mf1[0], (mf1[1] - mf2[1])/dt]
