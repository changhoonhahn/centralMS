'''





'''
import numpy as np 
from scipy.interpolate import interp1d

import util as UT 
import sfh as SFH
import observables as Obvs



class Evolver(object): 
    def __init__(self, PCH_catalog, theta, nsnap0=20): 
        ''' Using a given catalog of galaxy and subhalo snapshots track 
        galaxy star formation histories. 

        Parameters
        ----------

        PCH_catalog : (obj)
            Object that contains the subhalo accretion histories
    
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

    def Evolve(self): 
        ''' Evolve the galaxies from initial conditions specified in self.Initiate()
        '''
        # get integrated stellar masses 
        logM_integ = _Evolve_Wrapper(self.SH_catalog, self.nsnap0, 1,  
                theta_sfh=self.theta_sfh, 
                theta_sfms=self.theta_sfms, 
                theta_mass=self.theta_mass)

        # save into SH catalog
        isSF = np.where(self.SH_catalog['gclass'] == 'star-forming') 

        for ii, n_snap in enumerate(range(2, self.nsnap0)[::-1]): 
            self.SH_catalog['snapshot'+str(n_snap)+'_m.star'] = UT.replicate(
                    self.SH_catalog['m.sham'], 
                    len(self.SH_catalog['m.sham']))

            self.SH_catalog['snapshot'+str(n_snap)+'_m.star'][isSF] = logM_integ[ii]

        self.SH_catalog['m.star'] = UT.replicate(self.SH_catalog['m.sham'], len(self.SH_catalog['m.sham']))
        self.SH_catalog['m.star'][isSF] = logM_integ[-1]

        return None

    def Initiate(self): 
        ''' Assign the initial conditions to galaxies at z0. More specifically
        assign SFRs to galaxies at snpashot self.nsnap0 based on their SHAM 
        stellar masses, theta_gv, theta_sfms, and theta_fq. 

        Details 
        -------
        * Assign SFRs to galaxies *with* weights
        '''
        for i in range(2, self.nsnap0+1): # "m.star" from subhalo catalog is from SHAM
            self.SH_catalog['snapshot'+str(i)+'_m.sham'] = self.SH_catalog.pop('snapshot'+str(i)+'_m.star') 
        self.SH_catalog['m.sham'] = self.SH_catalog.pop('m.star')  

        keep = np.where(self.SH_catalog['weights'] > 0) # only galaxies that are weighted
    
        # assign SFRs at z0 
        sfr_out = assignSFRs(
                self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.sham'][keep], 
                np.repeat(UT.z_nsnap(self.nsnap0), len(keep[0])), 
                theta_GV = self.theta_gv, 
                theta_SFMS = self.theta_sfms,
                theta_FQ = self.theta_fq) 
    
        # save z0 SFR into self.SH_catalog 
        for key in sfr_out: 
            self.SH_catalog['snapshot'+str(self.nsnap0)+'_'+key.lower()] = \
                    UT.replicate(sfr_out[key], len(self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.sham']))
            self.SH_catalog['snapshot'+str(self.nsnap0)+'_'+key.lower()] = sfr_out[key][keep] 
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


def _Evolve_Wrapper(SHcat, nsnap0, nsnapf, **theta): 
    ''' Evolve galaxies that remain star-forming throughout the snapshots. 
    '''
    # parse theta 
    theta_mass = theta['theta_mass']
    theta_sfh = theta['theta_sfh']
    theta_sfms = theta['theta_sfms']

    # precompute z(t_cosmic) 
    z_table, t_table = UT.zt_table()     
    z_of_t = interp1d(t_table, z_table, kind='cubic') 
    
    # galaxies in the subhalo snapshots (SHcat) that are SF throughout 
    isSF = np.where(SHcat['gclass'] == 'star-forming') # only includes galaxies with w > 0 
    
    # logSFR(logM, z) function and keywords
    logSFR_logM_z, dlogmdt_kwargs = SFH.logSFR_wrapper(SHcat, isSF, theta_sfh=theta_sfh, theta_sfms=theta_sfms)

    # now solve M*, SFR ODE 
    dlogmdt_kwargs['logsfr_M_z'] = logSFR_logM_z 
    dlogmdt_kwargs['f_retain'] = theta_mass['f_retain']
    dlogmdt_kwargs['zoft'] = z_of_t

    if theta_mass['solver'] == 'rk4':     # RK4
        f_ode = SFH.ODE_RK4
    elif theta_mass['solver'] == 'euler': # Forward euler
        f_ode = SFH.ODE_Euler

    logM_integ = f_ode(
            SFH.dlogMdt,                    # dy/dt
            SHcat['snapshot'+str(nsnap0)+'_m.star'][isSF],              # logM0
            t_table[nsnapf:nsnap0][::-1],            # t_final 
            theta_mass['t_step'],   # time step
            **dlogmdt_kwargs) 

    return logM_integ 


def _pickSF(SHcat, nsnap0=20, theta_fq=None, theta_fpq=None): 
    ''' Take subhalo catalog and then based on P_Q(M_sham, z) determine, which
    galaxies quench or stay star-forming
    ''' 
    nsnap_quench = np.repeat(-999, len(SHcat['weights']))  # snapshot where galaxy quenches 
    gclass = SHcat['snapshot'+str(nsnap0)+'_gclass']
    
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


def assignSFRs(masses, zs, theta_GV=None, theta_SFMS=None, theta_FQ=None): 
    ''' Given stellar masses, zs, and parameters that describe the 
    green valley, SFMS, and FQ return SFRs

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

    # GV galaxy queiscent fraction 
    gv_fQ = qf.Calculate(mass=masses[isgreen], sfr=output['SFR'][isgreen], z=zs[isgreen], 
            mass_bins=np.arange(masses.min()-0.1, masses.max()+0.2, 0.2), theta_SFMS=theta_SFMS)
    
    # quiescent galaxies 
    fQ_gv = interp1d(gv_fQ[0], gv_fQ[1] * f_gv(gv_fQ[0]))
    
    isnotgreen = np.where(rand >= f_gv(masses))[0]
    fQ_true = qf.model(masses[isnotgreen], zs[isnotgreen], lit=theta_FQ['name']) - fQ_gv(masses[isnotgreen])
    isq = isnotgreen[np.where(rand[isnotgreen] < fQ_true)]
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


def defaultTheta(): 
    ''' Return generic default parameter values
    '''
    theta = {} 

    theta['gv'] = {'slope': 1.03, 'fidmass': 10.5, 'offset': -0.02}
    theta['sfms'] = {'name': 'linear', 'zslope': 1.14}
    theta['fq'] = {'name': 'cosmos_tinker'}
    theta['fpq'] = {'slope': -2.079703, 'offset': 1.6153725, 'fidmass': 10.5}
    theta['mass'] = {'solver': 'euler', 'f_retain': 0.6, 't_step': 0.1} 
    theta['sfh'] = {'name': 'constant_offset', 'nsnap0': 20}

    return theta 


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
