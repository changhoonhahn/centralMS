'''


Functions to measure observables of a given galaxy sample. 
The main observables are: 
    SFMS, SFM, SMHMR


'''
import util as UT 
import numpy as np 
# --- local --- 
from sham_hack import SMFClass 


def f_sat(logm, z): 
    ''' satellite fraction (Figure 3 b)
    parameterized as 

    f_sat = B0(logm) + B1(logm) * z 

    Source: Wetzel et al.(2013)
    '''
    if logm < 10.: 
        b0, b1 = 0.33, -0.055
    elif 10. <= logm < 10.5: 
        b0, b1 = 0.3, -0.073
    elif 10.5 <= logm < 11.: 
        b0, b1 = 0.25, -0.11
    elif 11. <= logm : 
        b0, b1 = 0.17, -0.1
    return b0 + b1 * z


def MF_data(source='li-white', m_arr=None):
    ''' Read in observed MFs
    '''
    if source == 'li-white': 
        f = ''.join([UT.dat_dir(), 'observations/li_white_2009.smf.dat'])
        mlow, mhigh, mmid, phi, err = np.loadtxt(f, unpack=True, usecols=[0,1,2,3,4]) 
        mlow -= 2.*np.log10(0.7)
        mhigh -= 2.*np.log10(0.7)
        mmid -= 2.*np.log10(0.7)
        phi *= 0.7**3
        err *= 0.7**3
        
        if m_arr is not None: # rebin 
            phi_arr = np.zeros(len(m_arr))
            err_arr = np.zeros(len(m_arr))
            for i, mm in enumerate(m_arr): 
                phi_arr[i] = phi[(np.abs(mmid - mm)).argmin()] 
                err_arr[i] = err[(np.abs(mmid - mm)).argmin()] 
        else: 
            m_arr = mmid
            phi_arr = phi
            err_arr = err
    return [m_arr, phi_arr, err_arr]


def getMF(masses, weights=None, m_arr=None, box=250, h=0.7): 
    ''' Calculate the Mass Function phi for a given set of masses.
    Return Phi(m_arr) 
    '''
    if not np.all(np.isfinite(masses)): 
        notfin = np.where(np.isfinite(masses) == False) 
        print masses[notfin], weights[notfin]
        #print np.sum(np.isfinite(masses)) 
        raise ValueError
    
    if weights is None: 
        w_arr = np.repeat(1.0, len(masses))
    else: 
        w_arr = weights

    if m_arr is None:  # by default assumes it's calculating SMF
        m_arr = np.arange(6.0, 12.1, 0.1) 

    # calculate d logM
    dm_arr = m_arr[1:] - m_arr[:-1]
    if np.abs(dm_arr - dm_arr[0]).max() > 0.001: 
        raise ValueError('m_arr has to be evenly spaced!')
    dlogm = dm_arr[0]

    # calculate logM bin edges
    mbin_edges = np.append(m_arr - 0.5*dlogm, m_arr[-1] + 0.5*dlogm) 

    Ngal,_ = np.histogram(masses, bins=mbin_edges, weights=w_arr) # number of galaxies in mass bin  

    vol = box ** 3  # box volume
    phi = Ngal.astype('float') / vol /dlogm * h**3

    return [m_arr, phi]


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
        if weights is None: # weights
            ws = np.repeat(1., len(mass))
        else: 
            ws = weights 
        if theta_SFMS is None: 
            raise ValueError
        if sfr_class is None: 
            if sfr is None or z is None: 
                raise ValueError
            # classify galaxies SF/Q  
            sfq = self.Classify(mass, sfr, z, weights=weights, theta_SFMS=theta_SFMS)
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

        f_q, count_arr = np.zeros(len(mass_mid)), np.zeros(len(mass_mid))
        for i_m in xrange(len(mass_mid)):
            masslim = np.where(
                    (mass > mass_low[i_m]) & 
                    (mass <= mass_high[i_m]))[0]
            ngal_mass = len(masslim)
            if ngal_mass == 0:  # no galaxy in mass bin 
                continue 
            
            isQ = masslim[np.where(sfq[masslim] == 'q')]
            f_q[i_m] = np.sum(ws[isQ])/np.sum(ws[masslim])
            count_arr[i_m] = np.sum(ws[masslim])
    
        if not counts: 
            return [mass_mid, f_q]
        else: 
            return [mass_mid, f_q, count_arr]
    
    def Classify(self, mstar, sfr, z_in, weights=None, theta_SFMS=None):
        ''' Classify galaxies based on M*, SFR, and redshift inputs.
        Returns an array of classifications
        '''
        ngal = len(mstar)
        if weights is None: 
            ws = np.repeat(1., ngal)
        else: 
            ws = weights
        hasw = np.where(ws > 0.)
        
        if isinstance(z_in, np.ndarray): 
            sfr_class = self.SFRcut(mstar[hasw], z_in[hasw], theta_SFMS=theta_SFMS)
        else: 
            sfr_class = self.SFRcut(mstar[hasw], z_in, theta_SFMS=theta_SFMS)

        sfq = np.empty(len(mstar), dtype=(str,16))

        sf_index = np.where(sfr[hasw] > sfr_class)
        sfq[hasw[0][sf_index]] = 'sf'
        q_index = np.where(sfr[hasw] <= sfr_class)
        sfq[hasw[0][q_index]] = 'q'

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
    
    def Calculate(self, mhalo, mstar, dmhalo=0.1, weights=None, bells=None, whistles=None):
        ''' 
        ''' 
        if weights is not None: 
            if len(mhalo) != len(weights): 
                raise ValueError('lenghts of mhalo and weights do not match!')

        m_low = np.arange(mhalo.min(), mhalo.max(), dmhalo)
        m_high = m_low + dmhalo

        mu_mstar = np.zeros(len(m_low)) 
        sig_mstar = np.zeros(len(m_low))
        counts = np.zeros(len(m_low))
        for i_m in range(len(m_low)):
            inbin = np.where((mhalo >= m_low[i_m]) & (mhalo < m_high[i_m]))
            
            if weights is None: 
                if len(inbin[0]) == 0: 
                    continue
                counts[i_m] = len(inbin[0])
                mu_mstar[i_m] = np.mean(mstar[inbin])
                sig_mstar[i_m] = np.std(mstar[inbin]) 
            else: 
                if np.sum(weights[inbin]) == 0.: 
                    continue
                counts[i_m] = np.sum(weights[inbin])
                mu_mstar[i_m] = np.average(mstar[inbin], weights=weights[inbin])
                sig_mstar[i_m] = np.sqrt(np.average((mstar[inbin]-mu_mstar[i_m])**2, weights=weights[inbin]) )
 
        return [0.5*(m_low + m_high), mu_mstar, sig_mstar, counts]

    def sigma_logMstar(self, mhalo, mstar, weights=None, Mhalo=12., dmhalo=0.1): 
        ''' Calculate sigma_logM* for a specific Mhalo. Default Mhalo is 10**12
        '''
        inbin = np.where((mhalo >= Mhalo-dmhalo) & (mhalo < Mhalo+dmhalo)) 
        if weights is not None: 
            mu_mstar = np.average(mstar[inbin], weights=weights[inbin])
            sig_mstar = np.sqrt(np.average((mstar[inbin]-mu_mstar)**2, weights=weights[inbin]))
        else: 
            mu_mstar = np.mean(mstar[inbin])
            sig_mstar = np.std(mstar[inbin])
        return sig_mstar


def SSFR_SFMS_obvs(mstar, z_in, lit='lee'): 
    ''' SSFR of SFMS derived from best-fit models of the observations 
    '''
    if lit == 'lee': # Lee et al. (2015) 
        if (z_in >= 0.25) and (z_in < 0.46): 
            S0 = 0.80 
            M0 = 10.03 
            gamma = 0.92
        elif (z_in >= 0.46) and (z_in < 0.63): 
            S0 = 0.99 
            M0 = 9.82 
            gamma = 1.13 
        elif (z_in >= 0.63) and (z_in < 0.78): 
            S0 = 1.23 
            M0 = 9.93 
            gamma = 1.11 
        elif (z_in >= 0.78) and (z_in < 0.93): 
            S0 = 1.35 
            M0 = 9.96 
            gamma = 1.28
        elif (z_in >= 0.93) and (z_in < 1.11): 
            S0 = 1.53 
            M0 = 10.10 
            gamma = 1.26 
        elif (z_in >= 1.11) and (z_in < 1.30): 
            S0 = 1.72 
            M0 = 10.31 
            gamma = 1.07 
        else: 
            raise ValueError

        return S0 - np.log10(1.+(10**(mstar-M0))**(-1.*gamma)) - mstar
    else: 
        raise NotImplementedError


if __name__=="__main__": 
    print MF_data(source='li-white', m_arr=np.linspace(9., 12., 20))
