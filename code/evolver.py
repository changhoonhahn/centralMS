'''





'''
import numpy as np 
import util as UT 
from scipy.interpolate import interp1d
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



    def Initiate(self): 
        ''' Assign the initial conditions to galaxies at z0. More specifically
        assign SFRs to galaxies at snpashot self.nsnap0 based on their SHAM 
        stellar masses, theta_gv, theta_sfms, and theta_fq. 

        Details 
        -------
        * Assign SFRs to galaxies *with* weights
        '''
        for i in range(1, self.nsnap0+1): # "m.star" from subhalo catalog is from SHAM
            self.SH_catalog['snapshot'+str(i)+'_m.sham'] = self.SH_catalog.pop('snapshot'+str(i)+'_m.star') 

        keep = np.where(self.SH_catalog['weights'] > 0) # only galaxies that are weighted
    
        # assign SFRs at z0 
        sfr_out = assignSFRs(
                self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.star'][keep], 
                np.repeat(UT.z_nsnap(self.nsnap0), len(keep[0])), 
                theta_GV = self.theta_gv, 
                theta_SFMS = self.theta_sfms,
                theta_FQ = self.theta_fq) 

        for key in sfr_out: 
            self.SH_catalog['snapshot'+str(self.nsnap0)+'_'+key.lower()] = \
                    UT.replicate(sfr_out[key], len(self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.star']))
            self.SH_catalog['snapshot'+str(self.nsnap0)+'_'+key.lower()] = sfr_out[key][keep]
       
       self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.star'] = self.SH_catalog['snapshot'+str(self.nsnap0)+'_m.sham']
        
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

        return None


def Evolve_a_Snapshot(SHcat, nsnap_i, **theta): 
    ''' Evolve the galaxies in SHcat from nsnap_i to nsnap_i-1. 
    Galaxies fall into 3 categories: quiescent, star-forming, 
    quenching. 
    '''
    dt = UT.t_nsnap(nsnap_i - 1) - UT.t_nsnap(nsnap_i)

    ssfrs_i = SHcat['snapshot'+str(nsnap_i)+'_sfr'] - SHcat['snapshot'+str(nsnap_i)+'_m.star']
    
    qf = Obvs.Fq()
    
    for key in SHcat.keys(): 
        if 'snapshot'+str(nsnap_i) in key: 
            SHcat[key.replace(str(nsnap_i), str(nsnap_i-1))] = UT.replicate(SHcat[key], len(SHcat[key]))

    ################
    # 1) quiescent galaxies.
    # M* given by SHAM; evolved to conserve SSFR 
    isQ = np.where(SHcat['snapshot'+str(nsnap_i)+'_gclass'] == 'quiescent')[0]
    # quiescent galaxies stay quiescent 
    SHcat['snapshot'+str(nsnap_i-1)+'_m.star'][isQ] = 'quiescent'
    # M_*(t_i+1) = M_sham(t_i+1)
    SHcat['snapshot'+str(nsnap_i-1)+'_m.star'][isQ] = SHcat['snapshot'+str(nsnap_i)+'_m.star'][isQ]
    # SFR = M_*(t_i+1) + SSFR(t_i)
    SHcat['snapshot'+str(nsnap_i-1)+'_sfr'][isQ] = ssfrs_i[isQ] + SHcat['snapshot'+str(nsnap_i)+'_m.star'][isQ]

    ################
    # 2) Star-forming galaxies
    isSF = np.where(SHcat['snapshot'+str(nsnap_i)+'_gclass'] == 'starforming')

    # quenching probability/Gyr
    mf = _SnapCat_mf(SHcat, nsnap_i, prop='m.sham')     # MF 
    dmf_dt = _SnapCat_dmfdt(SHcat, nsnap_i, prop='m.sham')  # dMF/dt
    mf_M = interp1d(mf[0], mf[1]) 
    dmf_dt_M = interp1d(mf_dt[0], mf_dt[1]) 

    # quenching probabily "fudge factor"
    fPQ_M = lambda mm: theta['fpq']['slope'] * (mm - theta['fpq']['fidmass']) + theta['fpq']['offset']

    # fPQ * ( dfQ/dt * MF + fQ * dMF/dt ) or fPQ * (Moustakas2013 quenching rate)
    Pq_M = lambda mm: fPQ_M(mm) * (
            qf.dfQ_dz(mm, zz, lit=theta['theta_fq']['name']) / UT.dt_dz(zz) * mf_M(mm) +
            qf.model(mm, zz, lit=theta['theta_fq']['name']) * dmf_dt_M(mm)
            )
    rand = np.random.uniform(0., 1., len(isSF[0])) 
    # Star-forming galaxies that start quenching between t_i and t_i+1
    now_qing = np.where(rand < Pq_M(SHcat['snapshot'+str(nsnap_i)+'_m.star'][isSF]) * dt) 
    tQs = np.random.uniform(UT.t_nsnap(nsnap_i - 1), UT.t_nsnap(nsnap_i)) # when SF galaxies start to quench

    # evolve M* and SFR of star-forming (t_i --> t_i+1 or t_i --> t_Q) 
    # some wrapper for all SFR(M*, t) prescriptions

    # now solve M*, SFR ODE 
    z_table, t_table = UT.zt_table()     
    z_of_t = interpolate.interp1d(list(reversed(t_table)), list(reversed(z_table)), kind='cubic') 
    func_kwargs = {
            't_initial': UT.t_nsnap(nsnap_i), 
            't_final': tf, 
            'f_retain': theta['mass']['f_retain'], 
            'zfromt': z_of_t, 
            'sfh_kwargs': sfh_kwargs
            }

    if theta['mass']['type'] == 'rk4':     # RK4
        f_ode = sfrs.ODE_RK4
    elif theta['mass']['type'] == 'euler': # Forward euler
        f_ode = sfrs.ODE_Euler
    integ_logM = f_ode(
            sfrs.dlogMdt_MS,                    # dy/dt
            SHcat['snapshot'+str(nsnap_i)+'_m.star'],              # logM0
            UT.t_nsnap(nsnap_i - 1),            # t_final 
            self.evol_dict['mass']['t_step'],   # time step
            **func_kwargs) 

    ################
    # 3) quenching galaxies: ones are in the process of quenching
    isQing = np.where(SHcat['snapshot'+str(nsnap_i)+'_gclass'] == 'qing')
    # SFR(t_i+1) = SFR(t_i) * exp(-dt / tauQ)
    sfr_qing = lambda sfr0, mq: sfr0 - 0.43429 * dt / Obvs.tauQ(mq, theta_tau=theta['theta_tau'])
    
    return SHcat


def pickSF(SHcat, nsnap0=20, nsnapf=1,**theta): 
    ''' Take subhalo catalog and then based on P_Q(M_sham, z) determine, which
    galaxies quench or stay star-forming
    '''
    z0_SF = np.where(SHcat['Gclass'] == 'star-forming')  # identify z0 SF galaxies 

    # then go from nsnap_0 --> nsnap_f and identify the quenching 
    # galaxies based on P_Q
    for n in range(nsnapf+1, nsnap0+1)[::-1]: 

        t_step = UT.t_nsnap(n - 1) - UT.t_nsnap(n) # Gyr between Snapshot n and n-1 
        
        # P_Q^cen = f_PQ * ( d( n_Q)/dt 1/n_SF ) 

        # calculate the necessary mass functions to estimate P_Q
        mf_Q_n = Obvs.getMF()

        mf_Q_n_1 = Obvs.getMF()

        # P_q_tot = PQ
        P_q_tot = PQ * t_step

        m_sham = SHcat['snapshot'+str(n)+'_m.sham'][z0_SF]
        


    

    return SHcat 


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
    output['Gclass'][isgreen] = 'qing'
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

    return theta 

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
    assert mf1[0] == mf2[0]

    return [mf1[0], (mf1[1] - mf2[1])/dt]
