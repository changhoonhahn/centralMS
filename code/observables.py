'''


Functions to measure observables of a given galaxy sample. 
The main observables are: 
    SFMS, SFM, SMHMR


'''
import numpy as np 

# --- local --- 
from sham_hack import SMFClass 


def getMF(masses, weights=None, m_arr=None, dlogm=0.1, box=250, h=0.7): 
    ''' Calculate the Mass Function for a given set of masses.
    '''
    if m_arr is None:  # by default assumes it's calculating SMF
        m_arr = np.arange(6.0, 12.1, dlogm) 

    if weights is None: 
        w_arr = np.repeat(1.0, len(masses))
    else: 
        w_arr = weights

    vol = box ** 3  # box volume
    
    if not np.all(np.isfinite(masses)): 
        print np.sum(np.isfinite(masses)) 
        raise ValueError
    
    Ngal, mbin_edges = np.histogram(masses, bins=m_arr, weights=w_arr) # number of galaxies in mass bin  

    mbin = 0.5 * (mbin_edges[:-1] + mbin_edges[1:]) 
    phi = Ngal.astype('float') / vol /dlogm * h**3

    return [mbin, phi]


def analyticSMF(redshift, m_arr=None, dlogm=0.1, source='li-drory-march'): 
    ''' Analytic SMF for a given redshift. 

    Return
    ------
    [masses, phi] : 
        array of masses, array of phi (smf number density) 
    '''
    if redshift < 0.1:
        redshift = 0.1
    if m_arr is None: 
        m_arr = np.arange(6.0, 12.1, dlogm) 

    MF = SMFClass(source=source, redshift=redshift)
    
    mass, phi = [], [] 
    for mi in m_arr: 
        if source in ('cool_ages', 'blanton'): 
            mass.append(mi - 0.5 * dlogm)
            phi.append(MF.numden(-mi, -mi+dlogm)/dlogm) 
        else: 
            mass.append(mi + 0.5 * dlogm)
            phi.append(MF.numden(mi, mi+dlogm)/dlogm) 
    #print 'Analytic ', np.sum(np.array(phi))
    return [np.array(mass), np.array(phi)]


class Fq(object): 
    def __init__(self, **kwargs): 
        ''' Class for quiescent fraction. Methods include calculating them, analytic models of them etc. 
        '''
        self.kwargs = kwargs
        # mass bin 
        mb = np.arange(9.0, 12.0, 0.2)

        self.mass_low  = mb[:-1]
        self.mass_high = mb[1:]
        self.mass_mid  = 0.5 * (self.mass_low + self.mass_high) 
    
    def Calculate(self, mass=None, sfr=None, z=None, weights=None, sfr_class=None, mass_bins=None, theta_SFMS=None, counts=False):
        ''' Calculate the quiescent fraction 
        '''
        # input cross-checks 
        if theta_SFMS is None: 
            raise ValueError
        if sfr_class is None: 
            if sfr is None or z is None: 
                raise ValueError
            sfq = self.Classify(mass, sfr, z, theta_SFMS=theta_SFMS)
        else: 
            sfq = sfr_class  
    
        if mass_bins is None: 
            mass_mid = self.mass_mid
            mass_low = self.mass_low
            mass_high = self.mass_high
        else: 
            mass_mid = 0.5 * (mass_bins[:-1] + mass_bins[1:])
            mass_low = mass_bins[:-1]
            mass_high = mass_bins[1:]

        if weights is None: 
            ws = np.repeat(1., len(sfq))
        else: 
            ws = weights 
            
        f_q = np.zeros(len(mass_mid)) 
        count_arr = np.zeros(len(mass_mid))
        for i_m in xrange(len(mass_mid)):
            masslim = np.where(
                    (mass > mass_low[i_m]) & 
                    (mass <= mass_high[i_m]))[0]
            ngal_mass = len(masslim)
            if ngal_mass == 0:  # no galaxy in mass bin 
                continue 
            
            isQ = masslim[np.where(sfq[masslim] == 'quiescent')]
            f_q[i_m] = np.sum(ws[isQ])/np.sum(ws[masslim])
            count_arr[i_m] = np.sum(ws[masslim])
    
        if not counts: 
            return [mass_mid, f_q]
        else: 
            return [mass_mid, f_q, count_arr]
    
    def Classify(self, mstar, sfr, z_in, theta_SFMS=None):
        ''' Classify galaxies based on M*, SFR, and redshift inputs.
        Returns an array of classifications
        '''
        sfr_class = self.SFRcut(mstar, z_in, theta_SFMS=theta_SFMS)

        sfq = np.empty(len(mstar), dtype=(str,16))

        sf_index = np.where(sfr > sfr_class)
        sfq[sf_index] = 'star-forming'
        q_index = np.where(sfr <= sfr_class)
        sfq[q_index] = 'quiescent'

        return sfq 

    def SFRcut(self, mstar, zin, theta_SFMS=None):
        ''' Specific SFR cut off used to classify SF or Quiescent 
        galaxies 
        ''' 
        #lowmass = np.where(mstar < 9.5)
        #factor = np.repeat(0.8, len(mstar))
        #factor[lowmass] = 1.0 
        #return -0.75 + 0.76*(zin-0.05) + 0.5*(mstar-10.5)
        #return -0.75 + 0.76*(zin-0.04) + factor*(mstar-9.5) - 0.8
        mu_sfr = SSFR_SFMS(mstar, zin, theta_SFMS=theta_SFMS) + mstar
        #offset = -0.75
        offset = -0.9
        return mu_sfr + offset

    def model(self, Mstar, z_in, lit='cosmos_tinker'):
        ''' Model quiescent fraction as a funcnction of 
        stellar mass and redshift from literature. Different methods 

        f_Q ( M_star, z) 

        Parameters
        ----------
        Mstar : array
            Galaxy stellar mass

        z_in : array
            Galaxy redshifts  

        lit : string
            String that specifies the model from literature 'cosmosinterp'
        '''

        if lit == 'cosmos_tinker': 
            qf_z0 = -6.04 + 0.63*Mstar
            
            try: 
                alpha = np.repeat(-2.57, len(Mstar))

                w2 = np.where((Mstar >= 10.) & (Mstar < 10.5))
                alpha[w2] = -2.52
                w3 = np.where((Mstar >= 10.5) & (Mstar < 11.))
                alpha[w3] = -1.47
                w4 = np.where((Mstar >= 11.) & (Mstar < 11.5))
                alpha[w4] = -0.55
                w5 = np.where(Mstar > 11.5)
                alpha[w5] = -0.12
            except TypeError: 
                if Mstar < 10.0: 
                    alpha = -2.57
                elif (Mstar >= 10.0) & (Mstar < 10.5): 
                    alpha = -2.52
                elif (Mstar >= 10.5) & (Mstar < 11.0): 
                    alpha = -1.47
                elif (Mstar >= 11.0) & (Mstar <= 11.5): 
                    alpha = -0.55
                elif (Mstar >= 11.5):
                    alpha = -0.12
            #else: 
            #    raise NameError('Mstar is out of range')

            output = qf_z0 * ( 1.0 + z_in )**alpha 
            try: 
                if output.min() < 0.0: 
                    output[np.where(output < 0.0)] = 0.0
                if output.max() > 1.0: 
                    output[np.where(output > 1.0)] = 1.0
            except TypeError:  
                if output < 0.0: 
                    output = 0.0
                elif output > 1.0: 
                    output = 1.0 

            return output 

        elif lit == 'cosmosinterp': 
            zbins = [0.36, 0.66, 0.88] 

            fq_z = [] 
            for zbin in zbins: 
                fq_file = ''.join([code_dir(), 'dat/wetzel_tree/', 
                    'qf_z', str(zbin), 'cen.dat' ]) 
               
                # read in mass and quiescent fraction
                mass, fq = np.loadtxt(fq_file, unpack=True, usecols=[0,1])  
                fq_z.append( np.interp(Mstar, mass, fq)[0] )   # interpolate to get fq(Mstar)
            interp_fq_z = interp1d(zbins, fq_z)#, kind='linear')#, fill_value='extrapolate') 
            if not isinstance(z_in, np.ndarray): 
                return interp_fq_z(np.array([z_in]))
            else: 
                return interp_fq_z(z_in)

        elif lit == 'cosmosfit': 
            zbins = [0.36, 0.66, 0.88] 
            exp_sigma = [1.1972271, 1.05830526, 0.9182575] 
            exp_sig = np.interp(z_in, zbins, exp_sigma) 
            output = np.exp( ( Mstar - 12.0 )/exp_sig)
            massive = np.where(Mstar > 12.0) 
            output[massive] = 1.0
            return output

        elif lit == 'wetzel':       # Wetzel et al. 2013
            qf_z0 = -6.04 + 0.63*Mstar
            
            try: 
                alpha = np.repeat(-2.3, len(Mstar))

                w1 = np.where((Mstar >= 9.5) & (Mstar < 10.0))
                alpha[w1] = -2.1
                w2 = np.where((Mstar >= 10.) & (Mstar < 10.5))
                alpha[w2] = -2.2
                w3 = np.where((Mstar >= 10.5) & (Mstar < 11.))
                alpha[w3] = -2.0
                w4 = np.where(Mstar >= 11.)
                alpha[w4] = -1.3
            except TypeError: 
                if Mstar < 9.5: 
                    alpha = -2.3
                elif (Mstar >= 9.5) & (Mstar < 10.0): 
                    alpha = -2.1
                elif (Mstar >= 10.0) & (Mstar < 10.5): 
                    alpha = -2.2
                elif (Mstar >= 10.5) & (Mstar < 11.0): 
                    alpha = -2.0
                elif (Mstar >= 11.0): # & (Mstar <= 11.5): 
                    alpha = -1.3
            #else: 
            #    raise NameError('Mstar is out of range')

            output = qf_z0 * ( 1.0 + z_in )**alpha 
            try: 
                if output.min() < 0.0: 
                    output[np.where(output < 0.0)] = 0.0
                if output.max() > 1.0: 
                    output[np.where(output > 1.0)] = 1.0
            except TypeError:  
                if output < 0.0: 
                    output = 0.0
                elif output > 1.0: 
                    output = 1.0 

            return output 
        
        elif lit == 'wetzelsmooth': 
            #qf_z0 = -6.04 + 0.63*Mstar
            qf_z0 = -6.04 + 0.64*Mstar
            alpha = -1.75

            output = qf_z0 * ( 1.0 + z_in )**alpha 
            try: 
                if output.min() < 0.0: 
                    output[np.where(output < 0.0)] = 0.0
                if output.max() > 1.0: 
                    output[np.where(output > 1.0)] = 1.0
            except TypeError: 
                if output < 0.0: 
                    output = 0.0
                if output > 1.0: 
                    output = 1.0

            return output 

        elif lit == 'wetzel_alternate': 
            fqall = lambda A, alpha, z: A*(1.+z)**alpha
            fsat = lambda B0, B1, z: B0 + B1 * z
            fqsat = lambda C0, C1, M: C0 + C1 * M

            M_arr = np.array([9.75, 10.25, 10.75, 11.25]) 
            
            A_arr = np.repeat(0.227, len(M_arr)) 
            alpha_arr = np.repeat(-2.1, len(M_arr))
            B0_arr = np.repeat(0.33, len(M_arr))
            B1_arr = np.repeat(-0.055, len(M_arr))
            C0_arr = np.repeat(-3.26, len(M_arr))
            C1_arr = np.repeat(0.38, len(M_arr))
            
            # mass ranges
            w1 = np.where(M_arr < 10.0)
            w2 = np.where((M_arr >= 10.) & (M_arr < 10.5))
            w3 = np.where((M_arr >= 10.5) & (M_arr < 11.))
            w4 = np.where(M_arr >= 11.)

            A_arr[w1] = 0.227
            A_arr[w2] = 0.471
            A_arr[w3] = 0.775
            A_arr[w4] = 0.957
            alpha_arr[w1] = -2.1
            alpha_arr[w2] = -2.2
            alpha_arr[w3] = -2.0
            alpha_arr[w4] = -1.3
            B0_arr[w1] = 0.33
            B0_arr[w2] = 0.30
            B0_arr[w3] = 0.25
            B0_arr[w4] = 0.17
            B1_arr[w1] = -0.055 
            B1_arr[w2] = -0.073 
            B1_arr[w3] = -0.11 
            B1_arr[w4] = 0.1 

            
            out_arr = (fqall(A_arr, alpha_arr, z_in) - fqsat(C0_arr, C1_arr, M_arr) * fsat(B0_arr, B1_arr, z_in)) / (1 - fsat(B0_arr, B1_arr, z_in))
            out_arr = np.array([0.0]+list(out_arr))
            M_arr = np.array([9.0]+list(M_arr))

            interp_out = interp1d(M_arr, out_arr)
            within = np.where((Mstar > M_arr.min()) & (Mstar < M_arr.max())) 
            output = np.repeat(1.0, len(Mstar))
            output[within] = interp_out(Mstar[within])
            geha_lim = np.where(Mstar < 9.5) 
            output[geha_lim] = 0.

            massive_lim = np.where(Mstar > M_arr.max())
            extrap = lambda mm: (out_arr[-1] - out_arr[-2])/(M_arr[-1] - M_arr[-2]) * (mm - M_arr[-1]) + out_arr[-1]
            output[massive_lim] = extrap(Mstar[massive_lim])

            return output 
        else: 
            raise NameError('Not yet coded') 

    def dfQ_dz(self, Mstar, z_in, dz=0.01, lit='cosmos_tinker'):
        ''' Estimate dfQ/dt(Mstar, z_in) using the central difference formula
        '''
        fq_ip1 = self.model(Mstar, z_in+dz, lit=lit)
        fq_im1 = self.model(Mstar, z_in-dz, lit=lit)

        return (fq_ip1 - fq_im1)/(2. * dz)


class Ssfr(object): 
    def __init__(self, **kwargs): 
        ''' Class object that describes the sSFR distribution of a 
        galaxy population
        '''
        self.kwargs = kwargs.copy()

        # mass bins 
        self.mass_bins = [[9.7, 10.1], [10.1, 10.5], [10.5, 10.9], [10.9, 11.3]]

        self.ssfr_range = [-13.0, -7.0]
        self.ssfr_nbin = 40
        
        self.ssfr_dist = None
        self.ssfr_bin_edges = None 
        self.ssfr_bin_mid = None 

    def Calculate(self, mass, ssfr, weights=None): 
        ''' Calculate the SSFR distribution for the four hardcoded mass bins 
        from the mass and ssfr values 
        '''
        if len(mass) != len(ssfr): 
            raise ValueError("mass and ssfr lengths do not match")
        if weights is not None: 
            if len(mass) != len(weights):
                raise ValueError("mass and weights lengths do not match")

        self.ssfr_dist = [] 
        self.ssfr_bin_mid = [] 
        self.ssfr_bin_edges = [] 

        # loop through the mass bins
        for i_m, mass_bin in enumerate(self.mass_bins): 
            mass_lim = np.where(
                    (mass >= mass_bin[0]) & 
                    (mass < mass_bin[1])
                    )
            n_bin = len(mass_lim[0])
            
            w_bin = None
            if weights is not None: 
                w_bin = weights[mass_lim]

            # calculate SSFR distribution  
            dist, bin_edges = np.histogram(
                    ssfr[mass_lim], 
                    range=self.ssfr_range, 
                    bins=self.ssfr_nbin, 
                    weights=w_bin,
                    normed=True)

            self.ssfr_dist.append(dist)
            self.ssfr_bin_mid.append(0.5 * (bin_edges[:-1] + bin_edges[1:]))
            self.ssfr_bin_edges.append(bin_edges)
        
        return [self.ssfr_bin_mid, self.ssfr_dist]


class Smhmr(object): 
    def __init__(self, **kwargs): 
        self.kwargs = kwargs.copy()
    
    def Calculate(self, mhalo, mstar, dmhalo=0.1, bells=None, whistles=None):
        ''' 
        ''' 
        m_low = np.arange(mhalo.min(), mhalo.max(), dmhalo)
        m_high = m_low + dmhalo

        mu_mstar = np.zeros(len(m_low)) 
        sig_mstar = np.zeros(len(m_low))
        counts = np.zeros(len(m_low))
        for i_m in range(len(m_low)):
            inbin = np.where((mhalo >= m_low[i_m]) & (mhalo < m_high[i_m]))
            
            counts[i_m] = len(inbin[0])
            mu_mstar[i_m] = np.mean(mstar[inbin])
            sig_mstar[i_m] = np.std(mstar[inbin]) 
    
        return [0.5*(m_low + m_high), mu_mstar, sig_mstar, counts]


def SSFR_Qpeak(mstar):  
    ''' Roughly the average of the log(SSFR) of the quiescent peak 
    of the SSFR distribution. This is designed to reproduce the 
    Brinchmann et al. (2004) SSFR limits.
    '''
    #return -0.4 * (mstar - 11.1) - 12.61
    return 0.4 * (mstar - 10.5) - 1.73 - mstar 


def sigSSFR_Qpeak(mstar):  
    ''' Scatter of the log(SSFR) quiescent peak of the SSFR distribution 
    '''
    return 0.18 


def SSFR_SFMS(mstar, z_in, theta_SFMS=None): 
    ''' Model for the average SSFR of the SFMS as a function of M* at redshift z_in.
    The model takes the functional form of 

    log(SFR) = A * log M* + B * z + C

    '''
    assert theta_SFMS is not None 

    if theta_SFMS['name'] == 'linear': 
        # mass slope
        A_highmass = theta_SFMS['mslope']#0.53
        A_lowmass = theta_SFMS['mslope']#0.53
        try: 
            mslope = np.repeat(A_highmass, len(mstar))
        except TypeError: 
            mstar = np.array([mstar])
            mslope = np.repeat(A_highmass, len(mstar))
        # z slope
        zslope = theta_SFMS['zslope']            # 0.76, 1.1
        # offset 
        offset = np.repeat(-0.11, len(mstar))

    elif theta_SFMS['name'] == 'kinked': # Kinked SFMS 
        # mass slope
        A_highmass = theta_SFMS['mslope_high'] #0.53 
        A_lowmass = theta_SFMS['mslope_low'] 
        try: 
            mslope = np.repeat(A_highmass, len(mstar))
        except TypeError: 
            mstar = np.array([mstar])
            mslope = np.repeat(A_highmass, len(mstar))
        lowmass = np.where(mstar < 9.5)
        mslope[lowmass] = A_lowmass
        # z slope
        zslope = theta_SFMS['zslope']            # 0.76, 1.1
        # offset
        offset = np.repeat(-0.11, len(mstar))
        offset[lowmass] += A_lowmass - A_highmass 

    mu_SSFR = (mslope * (mstar - 10.5) + zslope * (z_in-0.0502) + offset) - mstar
    return mu_SSFR


def sigSSFR_SFMS(mstar): #, z_in, theta_SFMS=None): 
    ''' Scatter of the SFMS logSFR as a function of M* and 
    redshift z_in. Hardcoded at 0.3 
    '''
    return 0.3 


def tauQ(mstar, theta_tau={'name': 'instant'}): 
    ''' Quenching efold models as a function of stellar mass
    '''
    type = theta_tau['name']

    if type == 'constant':      # constant tau 

        n_arr = len(mstar) 
        tau = np.array([0.5 for i in xrange(n_arr)]) 

    elif type == 'linear':      # lienar tau(mass) 

        tau = -(0.8 / 1.67) * ( mstar - 9.5) + 1.0
        #if np.min(tau) < 0.1: #    tau[ tau < 0.1 ] = 0.1
         
    elif type == 'instant':     # instant quenching 

        n_arr = len(mstar) 
        tau = np.array([0.001 for i in range(n_arr)]) 

    elif type == 'discrete': 
        # param will give 4 discrete tau at the center of mass bins 
        masses = np.array([9.75, 10.25, 10.75, 11.25]) 

        tau = np.interp(mstar, masses, param) 
        tau[ tau < 0.05 ] = 0.05

    elif type == 'linear': 
        # param will give slope and yint of pivoted tau line 
        tau = theta_tau['slope'] * (mstar - theta_tau['fid_mass']) + theta_tau['yint']
        try: 
            if np.min(tau) < 0.001: 
                tau[np.where( tau < 0.001 )] = 0.001
        except ValueError: 
            pass 

    elif type == 'satellite':   # quenching e-fold of satellite

        tau = -0.57 * ( mstar - 9.78) + 0.8
        if np.min(tau) < 0.001:     
            tau[np.where( tau < 0.001 )] = 0.001

    elif type == 'satellite_upper': 
        tau = -0.57 * ( mstar - 9.78) + 0.8 + 0.15
        if np.min(tau) < 0.001:     
            tau[np.where( tau < 0.001 )] = 0.001

    elif type == 'satellite_lower': 
        tau = -0.57 * ( mstar - 9.78) + 0.8 - 0.15
        if np.min(tau) < 0.001:     
            tau[np.where( tau < 0.001 )] = 0.001
    elif type == 'long':      # long quenching (for qa purposes)

        n_arr = len(mstar) 
        tau = np.array([2.5 for i in xrange(n_arr)]) 

    else: 
        raise NotImplementedError('asdf')

    return tau 
