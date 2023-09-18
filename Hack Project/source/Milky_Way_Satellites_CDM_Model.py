#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:23:02 2019

@author: Francis-Yan Cyr-Racine, University of New Mexico

TODO: 1. Need to improve the tidal evolution model (Impact on survival probability). # Summer 2023 - Kind of
TODO: 2. Need to improve galaxy-size model (half-light radius). # Summer 2023 - Not Done
TODO: 3. Need to implement velocity-dependent self-interaction cross section. # Summer 2023 - - Kind of
TODO: 4. Need to implement more general subhalo mass function to capture broader range of dark matter physics.
TODO: 5. Need to choose better redshift for accretion.
TODO: 6. add the enhanced stripping of stars in the presence of SIDM core (i.e. arXiv:1603.08919)

"""

import numpy as np
import pickle
from colossus.halo import profile_nfw, concentration
from colossus.cosmology import cosmology
from scipy.special import hyp2f1, gammaln
from scipy.interpolate import interp1d
from scipy.stats import norm, poisson, skewnorm
from scipy.integrate import quad
import kalepy as kale
from astropy import constants as const
from astropy import units as u
from f_bound_Tidal_Evolution import MmxbyMmx0_2D_interp_Tmx_fixed, MmxbyMmx0_2D_interp_rcore_fixed, MmxbyMmx0_1D_interp_rcore_Tmx_fixed
from MWS_real_data_distribution import kernel_z_infall, z_infall_real_data, radii_KDE, R_apoc_real_data, R_peribyR_apoc_ratio, Halo_r12_real_data

# Load the simulation halo data (now only using survival probability from sims)
with open('../halo_sat_properties/halo_data.pkl', 'rb') as ff:
    halo_data = pickle.load(ff, encoding='latin1')

# Subhalo disruption probability due to Galactic disk from ML algorithm
disr_prob_1 = halo_data[30]['Halo_ML_prob']
surv_prob_1 = 1.0 - disr_prob_1
def surv_prob(f):
    
    # Subhalo data of 31st simulation from halo_data file 
    Halo_subs = halo_data[30]['Halo_subs']
    # Virial Mass at z = 0
    Halo_m_vir = np.array([Halo_subs[i][2] for i in range(Halo_subs.size)]) # units in M_sun
    # Mass at v_peak at infall
    Halo_m_peak = np.array([Halo_subs[i][11] for i in range(Halo_subs.size)]) # units in M_sun
    
    # Observational cut by considering M_min = 10**8 M_sun
    Halo_m_vir_cut = Halo_m_vir[Halo_m_vir >= 10**8.0] # units in M_sun
    Halo_m_peak_cut = Halo_m_peak[Halo_m_vir >= 10**8.0] # units in M_sun
    surv_prob_cut = surv_prob_1[Halo_m_vir >= 10**8.0]
    # Bound remnant of Subhalo
    Halo_f_bound = Halo_m_vir_cut/Halo_m_peak_cut
    
    # Create an 1-D interpolation between 'bound remnant' and 'survival probability'
    surv_prob_interp = interp1d(Halo_f_bound, surv_prob_cut)
    
    out = np.zeros_like(f)
    # Sort out if the subhaloes are completely disrupted or not
    disrupted = f <= 0.01
    survived = np.logical_not(disrupted)
    
    # Probility of Disruption
    out[disrupted] = 0.0 * f[disrupted]
    
    # Probility of Survival
    out[survived] = surv_prob_interp(f[survived])
    
    return out

# Load the galaxy-halo connection model
with open('../halo_sat_properties/interpolator.pkl', 'rb') as ff:
    vpeak_Mr_interp = pickle.load(ff, encoding='latin1')

# Set cosmology
cosmo = cosmology.setCosmology('planck18')

# Cosmological parameters
omega_b = cosmo.Ob0
omega_m = cosmo.Om0
h = cosmo.h

#Load the standard CDM mass concentration relation
mass_def_colossus = 'vir'
#z_conc = 0.5
# Milky-Way Circular Velogity 
V_MW = 220.0 # units in km/s (Check data Zhou et al. (2023) (https://iopscience.iop.org/article/10.3847/1538-4357/acadd9/pdf))
# Conversion factor from kpc/(km/s) to Gyrs
fac = 0.9777922216807892

# Infall Redshift of MW Satellites
def z_infall(s):
    
    if s == 1:
        out = np.array([kernel_z_infall.resample(size = s)]) #reflect = [z_infall_real_data.min(), z_infall_real_data.max()]
    else:
        out = kernel_z_infall.resample(size = s) #reflect = [z_infall_real_data.min(), z_infall_real_data.max()]
    
    return out

# Peri and Apoc Radii of MW Satellites at Infall
def joint_radii(s):
    R_apoc_resample, R_peribyR_apoc_ratio_resample = radii_KDE.resample(size = s, reflect = [[R_apoc_real_data.min(),                                                                  R_apoc_real_data.max()], [R_peribyR_apoc_ratio.min(),                                                                            R_peribyR_apoc_ratio.max()]])
    R_peri_resample = R_apoc_resample * R_peribyR_apoc_ratio_resample
    
    return R_peri_resample, R_apoc_resample

# TODO: Define a better T_orb function from given potential
# Galpy.orbit, input = RA, Dec, distance, proper motion, radial velocity
# Don't consider Dark Matter, since point mass in respect of MW
def Torb_infall(r_p, r_a): # units in Gyrs
    ratio = r_p/r_a
    a_sem = 0.5 * (r_p + r_a)
    out = np.zeros_like(r_p)
    
    # Sort out if we are in the circular orbit or elliptical orbit
    circular_orbit = ratio >= 0.8
    elliptical_orbit = np.logical_not(circular_orbit)
    
    # Orbital Period of Circular Orbit
    out[circular_orbit] = 2*np.pi*r_p[circular_orbit]*fac/V_MW
    
    # Orbital Period of Elliptical Orbit
    out[elliptical_orbit] = 2*np.pi*fac*a_sem[elliptical_orbit]*0.75/V_MW # 0.75 is forcefully multiplied to match the value with 
                                                                          # the paper Errani et al. (2023)
    
    return out

# The circular orbit time at radius r_max at infall or "characteristic crossing time"
def T_max(r, v):
    out = (2*np.pi*r*fac)/v # units in Gyrs
    return out

# The circular orbital time of subhaloes at pericentre at infall
def T_peri(r):
    out = (2*np.pi*r*fac)/V_MW # units in Gyrs
    return out

def bryan_norman_threshold(z):
    
    x = cosmo.Om0*(1+z)**3/cosmo.Ez(z)**2 - 1.0
    
    return 18*np.pi**2 + 82*x - 39*x**2


def prep_data(filename, MV_b, Sig_b):

    """
    Routine to prepare the binned data for likelihood analysis
    """

    # open file
    gal_pop = np.genfromtxt(filename).T

    Mr = gal_pop[7]
    sigma_star = gal_pop[6]
    Prob_surv = gal_pop[5]

    # Binned the data
    data, x, y = np.histogram2d(Mr,sigma_star,bins=[MV_b,Sig_b],weights=Prob_surv,normed=False)
    #data, x, y = np.histogram2d(Mr,sigma_star,bins=[MV_b,Sig_b],weights=Prob_surv)

    return data

def hyper_func(a,b,x):
    return hyp2f1(b,1-a+b,2-a+b,-x)

###############################################################################
# Subhalo Concentration routines
###############################################################################

# Let load the concentration correction from Dunstan et al. (arXiv:1109.6291)
filename = '../halo_sat_properties/WDM_alpha_conc.dat'
alpha_c = np.genfromtxt(filename)

# Extract the limit points where we reach the CDM limit
alpha_c_lim = alpha_c[-1,1]
m_lim = alpha_c[-1,0]

# Define interpolating function
alpha_c_int = interp1d(alpha_c[:,0],alpha_c[:,1])

# Define the \Delta\alpha_c function
def delta_alpha_c(m_WDM):
    """
    WDM correction function for the mass concentration relation
    """
    if m_WDM > m_lim:
        return 0
    else:
        return alpha_c_lim - alpha_c_int(m_WDM)
    
def c_vir(M,z):
    """
    Concentration-mass relation, using Colossus with the Ishiyama (2021) model.

    Parameters
    ----------
    M : Array
        Peak/Infall virial masses in M_sun/h units. 
    z : float
        Redshift (Infall). Default value is z_conc

    Returns
    -------
    conc: array
        The halo concentrations (at Infall).

    """    
    conc = concentration.concentration(M * h, mass_def_colossus, z, model='ishiyama21') 
    return conc

# Define the WDM mass concentration relation
def c_WDM(M,z,m_WDM):
    """ WDM mass concentration relation """
    cWDM = c_vir(M,z)*(M/3.26e12)**delta_alpha_c(m_WDM)
    return cWDM

def sigma_c(M,sigma_c_0=0.1,zeta=-0.33):
    """
    Scatter of the mass-concentration relation
    """
    return sigma_c_0*(M/1e9)**(zeta)

###############################################################################
# SIDM core density relations
###############################################################################

#conv_factor = 365.25*24*60*60*(1e-2)**2*1.989e33*1e3/(3.0856e16)**3
cov_factor = (u.Gyr*u.M_sun*u.kpc**(-3)*u.cm**2*u.g**(-1)*u.km*u.s**(-1)).to(u.dimensionless_unscaled)
G_Newt = 4.3009173*10**(-6) #in kpc (km/s)^2/M_sun
t_age = 13.78620642 #Age of the Milky Way in Gyrs


def r_core(x, logxlim=1.341, fac=2.07, fudge=1.263):
    """
    Function to compute the SIDM core size as a function of t/t_0
    
    Parameters
    ----------
    x : array, float
        The value of t/t_0 as defined in Nishikawa et al. (2020)
    logxlim : float
        Value of log10(x) where core-expansion is maximum. Default value is 1.341. 
    fac : float
        Factor to convert from the \beta\sigma t parameter in Yang et al. (2022) 
        to t/t_0 from Nishikawa et al. (2020). Default value if 2.07, but set to 1 
        to have the input be \beta\sigma t.
    fudge : float
        Fudge factor to account for the different definition of a "core" from
        Yang et al. (2022) compared to arXiv:2204.06568.
    
    Returns
    -------
    r_core : array,float
        The halo core size, in units of the NFW scale radius.
        
    """
    # Parameters from the fit from Yang et al. (2022), arXiv:2205.02957
    A1 = -0.1078
    B1 = 0.3737
    C1 = -0.7720
    A2 = -0.04049
    C2 = 43.07
    D2 = -43.07
    E = 2.238

    # Allow for both single entry or numpy arrays
    if isinstance(x,float) or isinstance(x,int):
        size = 1
        x = np.array([x])
    else:
        size = x.size
    out = np.zeros(size)
    
    # Sort out if we are in the core expansion or core collapse phase
    core_exp = np.log10(x/fac) <  logxlim
    core_coll = np.logical_and(np.logical_not(core_exp),np.log10(x/fac) < E)
    core_coll_lim = np.logical_and(np.logical_not(core_exp),np.log10(x/fac) >= E)
    
    # Core-expansion phase
    out[core_exp] = fudge*10**(A1*np.log10(x[core_exp]/fac)**2 + B1*np.log10(x[core_exp]/fac) + C1)
    
    # Core-collapse phase
    out[core_coll] = 10**(A2*(np.log10(x[core_coll]/fac) - (E+2))**2 + C2 + D2/(E + 0.0001 - np.log10(x[core_coll]/fac))**0.005)
    
    # Core-collapse limit
    out[core_coll_lim] = 10**(A2*(E - (E+2))**2 + C2 + D2/(E + 0.0001 - E)**0.005)
    
    return out  

def rho_core(x, fac=2.07):
    """
    Function to compute the SIDM core density as a function of t/t_0.
    Technically, this function is only valid in the core-collapse phase,
    but does seem to do a good job in all regime. 
    
    Parameters
    ----------
    x : float
        The value of t/t_0 as defined in Nishikawa et al. (2020)
    fac : float
        Factor to convert from the \beta\sigma t parameter in Yang et al. (2022) 
        to t/t_0 from Nishikawa et al. (2020). Default value if 2.07, but set to 1 
        to have the input be \beta\sigma t.
        
    Returns
    -------
    rho_core : array, float
        The central density of the halo.
    
    """
    # Parameters from the fit from Yang et al. (2022), arXiv:2205.02957
    E = 2.238
    A = 0.05771
    C = -21.64
    D = 21.11
    
    # Allow for both single entry or numpy arrays
    if isinstance(x,float) or isinstance(x,int):
        x = np.array([x])

    
    # Core density evolution 
    out = 10**(A*(np.log10(x/fac).clip(-2,E) - (E+3))**2 + C + D/(E + 0.0001 - np.log10(x/fac).clip(-2,E))**0.02)
   
    return out

def r_out(x, fac=2.07):
    """

    Function to trace the radius where the density profile
    influenced by self-interaction joins smoothly to the NFW outskirts.

    Parameters
    -----------
    x : float
        The value of t/t_0 as defined in Nishikawa et al. (2020)
    fac : float
        Factor to convert from the \beta\sigma t parameter in Yang et al. (2022) 
        to t/t_0 from Nishikawa et al. (2020). Default value if 2.07, but set to 1 
        to have the input be \beta\sigma t.
    
    Returns
    --------
    r_out : array,float
       The outer radius of the profile where it joins the NFW halo.

    """
    # Parameters from the fit from Yang et al. (2022), arXiv:2205.02957
    E = 2.238
    A3 = 0.02403
    C3 = -4.724
    D3 = 5.011

    # Core Density Self Interaction Radius
    if np.log10(x/fac) < E:
        out = 10**(A3*(np.log10(x/fac) - (E+2))**2 + C3 + D3/(E + 0.04 - np.log10(x/fac))**0.005)
    else:
        out = 10**(A3*(E - (E+2))**2 + C3 + D3/(E + 0.04 - E)**0.005)
    return out

def rho_hat(r_hat, x, logxlim=1.341, fac=2.07):
    """
    Function to define the halo density profile as described in Yang et al. (2022), arXiv:2205.02957.
    It is described by the NFW profile at the early stage of the halo evolution,
    and later described by triple power law at the later phases of halo evolution.

    Parameters
    -----------
    r_hat : array
        r_hat is defined as r/r_s.
    x : float
        The value of t/t_0 as defined in Nishikawa et al. (2020)      
    logxlim : float
        Value of log10(x) where core-expansion is maximum. Default value is 1.341.      
    fac : float
        Factor to convert from the \beta\sigma t parameter in Yang et al. (2022) 
        to t/t_0 from Nishikawa et al. (2020). Default value if 2.07, but set to 1 
        to have the input be \beta\sigma t.
    
    Returns
    --------
    rho_hat : array, float
         The dimensionless halo density profile, rho/rho_s. 

    """
    # Parameters from Yang et al. (2022), arXiv:2205.02957
    s = 2.19

    # Sort out if we are in the core expansion or core collapse phase
    if np.log10(x/fac) < logxlim: 
        # Halo density profile at the early stages of halo evolution
        out =  np.tanh(r_hat/r_core(x))/(r_hat*(1+r_hat)**2)
    else:
        # Halo density profile at the later phase of halo evolution
        out = rho_core(x)/((1 + r_hat/r_core(x))**s * (1 + r_hat/r_out(x))**(3-s))

    return out

def M_half_dimless(x,r_half_hat):
    """
    Function to compute the mass within the half-light radius of the SIDM 
    density profile.
    
    Parameters
    -----------
    x : float or array
        The value of t/t_0 as defined in Nishikawa et al. (2020)  
    r_half_hat : float or array
        r_half_hat is defined as r_half/r_s.
        
    Returns
    --------    
    dimless_mass : float
        The dimensionless mass within the half-light radius. Needs to be multiplied 
        by 4*pi*rho_s*r_s**3 to be converted to a physical mass. 
    
    """
    
    def integrand(r,x):
        return r**2*rho_hat(r,x)
    
    if isinstance(x,float) or isinstance(x,int):
       x = np.array([x])
       r_half_hat = np.array([r_half_hat])
   
    dimless_mass = np.array([quad(integrand,0,r_half_hat[i],args=(x[i]))[0] for i in range(len(x))])
    
    return dimless_mass

def sigma_v_o_m(v, sigma_0_o_m_c = 150.0, v_SIDM_c = 50.0, delta_SIDM = 2.0, beta_SIDM = 2.0):
    """
    Function to compute viscosity cross section of SIDM Subhaloes.
    
    Parameters
    -----------
    v : float or array
        The value of relative velocity of dark matter particles.
        Here we consider v_peak as v.
    sigma_0_o_m_c : 150.0 cm^2/g, the value of the normalization.
    v_SIDM_c : 50.0 km/s, the value of the the velocity scale.
    
    Returns
    --------
    sigma_v_o_m : float or array
                  Velocity Dependent dark matter self-interaction cross section over mass, in units of cm^2/g.
    
    """
    
    out = sigma_0_o_m_c / ((1 + (v/v_SIDM_c)**delta_SIDM)**beta_SIDM)
    
    return out
    
###############################################################################
# Mass loss and tidal track routine
###############################################################################

def bound_remnant(r, t_orb, T_idc):
    """
    Bound remnant of the MW subhaloes. The interpolations functions are derived 
    from Errai et. al. (2023) (https://doi.org/10.1093/mnras/stac3499).

    Parameters
    ----------
    r : r_core: array,float
        The halo core size, in units of the NFW scale radius.
    t_orb: array,float
           The number of orbits have been completed by the MW Subhaloes since accretion.
    T_idc: array,float
           The initial density contrast of the MW Subhaloes in logarithmic base 10.
    Returns
    -------
    bound_remnant: array,float
                   The bound remnant of the MW Subhaloes

    """
    bound_remnant = (MmxbyMmx0_2D_interp_Tmx_fixed((r, t_orb))*(MmxbyMmx0_2D_interp_rcore_fixed((T_idc, t_orb))/MmxbyMmx0_1D_interp_rcore_Tmx_fixed(t_orb))).clip(-2.5, 0.0)
    
    return bound_remnant

def tidal_tracks(fb):
    """
    Tidal tracks from Errani & Navarro (arXiv:2111.05866), also using some results
    from Penarrubia et al. (2010, arXiv:1002.3376) to convert from f_bound to 
    r_max/r_max,0.

    Parameters
    ----------
    fb : float
        Bound mass fraction left after tidal evolution.

    Returns
    -------
    r_fact : float
        r_max/r_max,0.
    v_fact : float
        v_max/v_max,0.

    """
    mu = -0.6
    eta = 0.40
    alpha = 0.4
    beta = 0.65
    r_fact = (2**mu*fb**eta)/(1+fb)**mu
    v_fact = 2**alpha*r_fact**beta*(1+r_fact**2)**(-alpha)
    return r_fact,v_fact


def r_cut_o_rmax0(fb):
    """
    Tidal cutoff radius in units of r_max,0 from Errani & Navarro 
    (arXiv:2011.07077), as a function of the bound mass fraction

    Parameters
    ----------
    fb : float
        Bound mass fraction remaining after tidal interaction.

    Returns
    -------
    rcut_o_rmax0 : float
        The ratio of the cutoff radius and of the initial r_max value.

    """
    a = 0.44
    b = 0.3
    c = -1.1
    
    rcut_o_rmax0 = a*fb**a*(1.0 - np.array(fb).clip(0,0.9999)**b)**c
    
    return rcut_o_rmax0

###############################################################################
# Core functions to generate MW satellite populations
###############################################################################

def gen_subhalo_pop(m_WDM, sigma_0_o_m_c = 150.0, v_SIDM_c = 50.0, M_min=10**5.0, mulim=32, Mrlim=0, alpha=1.87, beta=1.16,
                    K0=1.88e-3, M_host=1.17e12, sigma_r=0.20, alpha_m=-1.31, sigma_m=0.005, A=0.020,
                    n=1.5, sigma_c_0=0.1, zeta=-0.33, full_out=False):
    """
    Generate a mock MW satellite population
    
    Here we use the subhalo mass function from Dooley et al. (2017) (arXiv:1610.00708) 
    coupled with the WDM correction from Schneider et al. (2012) (arXiv:1112.0330)
    dN/dM_sub = K0*M_host*(M_sub)**(-alpha)*(1 + M_hm/M_sub)**(-beta),
    where M_hm is the half mode mass
    
    Parameters
    ----------
    
    m_WDM: warm dark matter mass in keV
    sigma_0_o_m_c : The normalization factor in the velocity dependent dark matter 
                    self-interaction cross section over mass, in units of cm^2/g.
    v_SIDM_c : The velocity scale above which the sigma_v_o_m falls off as v^-4, in units of km/s.
    M_min: minimum halo mass to be considered
    mulim: surface brightness limit beyond which we can't detect satellites
    Mrlim: magnitude limit beyond we can't detect satellites
    alpha: subhalo mass function slope 
    beta: parameter determining the shape of the WDM cutoff
    K0: normalization of the subhalo mass function
    M_host: mass of the Milky Way
    sigma_r: scatter for the half-light radius
    alpha_m: faint-end slope of the satellite luminosity function
    sigma_m: lognormal scatter of the satellite luminosity at fixed v_peak
    A: Normalization of the half light-radius relation (in kpc)
    n: slope of the half-light radius relation
    sigma_c_0: Amplitude of the scatter of the mass-concentration relation, normalized to M = 1e9M_sun
    zeta: Slope of the scatter of the mass-concentration relation
    full_out: whether to ouput the complete or partial output
        
    Returns
    -------
    
    """
    # Compute half-mode mass (we fix here the Bullock & Bolan-Kolchin bug)
    Mhm = 1.887e10*(omega_m*h**2/0.14)*((omega_m-omega_b)*h**2/0.1225)**0.33*(m_WDM)**(-3.33)

    # Maximum subhalo mass (here half the host mass)
    M_max = 10**11

    # Total number of subhalos (analytical integral of the mass function given above)
    N_sub = K0*M_host*Mhm**(-beta)*(1/(1-alpha+beta))*(M_max**(1-alpha+beta)*hyper_func(alpha,beta,M_max/Mhm) - M_min**(1-alpha+beta)*hyper_func(alpha,beta,M_min/Mhm))
    
    if N_sub <= 1.0:
        print(N_sub)
        print(m_WDM, sigma_0_o_m_c, v_SIDM_c, M_min, mulim, Mrlim, alpha, beta, K0, M_host, sigma_r, alpha_m, sigma_m, A, n)    
  
    # Compute CDF for WDM mass function
    # TODO: Derive CDF
    mvec = np.linspace(np.log10(M_min),np.log10(M_max),1000)
    CDF = ((10**mvec)**(1-alpha+beta)*hyper_func(alpha,beta,(10**mvec)/Mhm) - M_min**(1-alpha+beta)*hyper_func(alpha,beta,M_min/Mhm))/(M_max**(1-alpha+beta)*hyper_func(alpha,beta,M_max/Mhm) - M_min**(1-alpha+beta)*hyper_func(alpha,beta,M_min/Mhm))

    #Create interpolation function
    M_int = interp1d(CDF,mvec)

    # Actual number of subhalos in the realization (assume Poisson distributed)
    N_real = max(poisson.rvs(N_sub),1)
    
    # Get subhalo masses
    rand_int = np.random.random(N_real)
    log_M_peak = M_int(rand_int)
    
    # Dimension Size
    dim_size = log_M_peak.size
    
    # Infall Redshift
    zinfall = z_infall(dim_size)

    # Get concentration at peak
    cWDM = c_WDM(10**log_M_peak,zinfall,m_WDM)
    log_c_mean =  np.log(cWDM)
    c_peak = np.random.lognormal(log_c_mean,np.log(10)*sigma_c(10**log_M_peak,sigma_c_0,zeta))

    # Obtain NFW parameter given mass and concentration at peak
    rhos, rs = profile_nfw.NFWProfile.nativeParameters(M=10**log_M_peak*h, c=c_peak, z=zinfall, mdef=mass_def_colossus)
    # x_max = r_max/rs, where r_max is the radius at which the velocity curve peaks
    x_max = 2.1626 
    # Mass within the r_max radius 
    Mmax = 4*np.pi*rhos*rs**3*(np.log(1 + x_max) - x_max/(1+x_max))/h
    # Get vpeak
    vpeak = np.sqrt(G_Newt*Mmax*h/(x_max*rs))
    
    # The peri and apoc radii of Subhaloes at infall
    r_peri, r_apoc = joint_radii(zinfall.size) # # units in kpc

    # Get Rvir at peak (in kpc)
    rvir = c_peak*rs/h

    # half-light radii
    # TODO: Need a model for tidal heating of stars and the expansion of the half-light radius due to SIDM core formation (i.e. arXiv:1603.08919)
    Halo_r12_mean = A * (rvir/10)**n #Josh Paper? #Real data from Simon # Compare # Relation between infall and today
    Halo_r12 = np.random.lognormal(np.log(Halo_r12_mean),np.log(10)*sigma_r).clip(Halo_r12_real_data.min(), Halo_r12_real_data.max())

    # Sort and compute absolute magnitude of satellites using Ethan Nadler's galaxy-halo connection
    # TODO: add the enhanced stripping of stars in the presence of SIDM core (i.e. arXiv:1603.08919)
    idx = np.argsort(np.argsort(vpeak))
    #Mr = (np.log(10)*sigma_m*np.random.randn(vpeak.size) + vpeak_Mr_interp(vpeak, alpha_m))[idx]
    
    # Mean absolute magnitude of satellites
    Mr_mean = vpeak_Mr_interp(vpeak, alpha_m)[idx]
    # Absolute magnitude of Sun
    Mbol_sun = 4.81
    # Conversion from mean absolute magnitude to mean luminosity of satellites
    L_mean = 10**((-1. * Mr_mean + Mbol_sun)/2.5) # units in L_sun
    # luminosity of satellites with lognormal scatter of the satellite luminosity at fixed v_peak
    Lum = np.random.lognormal(np.log(L_mean),(np.log(10)*sigma_m)) # units in L_sun
    # Conversion from luminosity to absolute magnitude of satellites
    Mr = -1. * (2.5 * (np.log10(Lum)) - Mbol_sun)
    
    # Surface brightness and magnitude cut
    #M_tot = mulim - 36.57 - 2.5*np.log10(2*np.pi*(Halo_r12[idx]**2))
    #idx_s = (Mr < M_tot) & (Mr < Mrlim)
    
    #Keep only those passing the observational cuts
    #rs = rs[idx_s]
    #rhos = rhos[idx_s]
    #Halo_r12 = Halo_r12[idx_s]
    #zinfall = zinfall[idx_s]
    #vpeak = vpeak[idx_s]
    #r_peri = r_peri[idx_s]
    #r_apoc = r_apoc[idx_s]
    
    #Number of observed satellites
    N_obs = rs.size
    
    # Fractional half-light radius
    x12 = h*Halo_r12/rs

    ### Compute sigma_*, the stellar velocity dispersion ###
    # initialize the array
    sigmastar = np.zeros(N_obs)
    M12 = np.zeros(N_obs)
    
    # Randomly assign tidal mass loss using a skewnorm distribution. This was roughly
    # calibrated from sims, it peaks around 0.3 with a long tail towards 1. 
    # TODO: This should be improved to be a function of concentration
    
    # The infall time of Subhaloes
    t_infall = cosmology.Cosmology.lookbackTime(cosmo, z = zinfall) # units in Gyrs
    
    # The orbital period of Subhaloes at infall
    Torbinfall = Torb_infall(r_peri, r_apoc) # units in Gyrs
    
    # The total number of orbits completed by the subhaloes till now
    num_total_orbits = t_infall/Torbinfall
    
    # Velocity Dependent Dark matter self-interaction cross section over mass, in units of cm^2/g
    sigma_o_m = sigma_v_o_m(vpeak, sigma_0_o_m_c, v_SIDM_c) # units in cm^2/g
    
    # Compute the typical self-interaction time scale at peak
    t_over_t0 = t_infall * cov_factor * h**2 * sigma_o_m * rhos * np.sqrt(4*np.pi*G_Newt*rs**2*rhos) # look into t_age
    
    # Cored Radius of Subhaloes at infall
    rcore = r_core(t_over_t0)
    
    # The circular orbital time of subhaloes at pericentre at infall
    Tperi = T_peri(r_peri) # units in Gyrs
    
    # The period of a circular orbit of radius r_max at infall
    r_max = x_max*rs/h # units in kpc
    Tmax = T_max(r_max, vpeak) # units in Gyrs
    
    # Initial Density Contrast of Subhaloes or Initial Cr
    log10_T_maxbyT_peri = np.log10(Tmax/Tperi).clip(-0.52, 0.03)
    
    # Tidal Mass Loss of Subhaloes
    log10_f_bound = bound_remnant(rcore, num_total_orbits, log10_T_maxbyT_peri)
    #f_bound = skewnorm.rvs(5, 0.15,0.5, size=N_obs).clip(0.05,1)
    rfact, vfact = tidal_tracks(10**log10_f_bound)
    
    # Tidal truncation radius (in kpc/h)
    r_cut = r_cut_o_rmax0(10**log10_f_bound)*rs*x_max
    
    # Survival probability
    Prob_s_1 = 1.0 - np.random.choice(disr_prob_1,N_real)
    Prob_s = surv_prob(10**log10_f_bound)
    
    # Determine whether the halos are NFW 
    nfw = t_over_t0  < 0.01
    # If NFW compute mass within half-light radius
    # TODO: check whether we always have r_12 < r_cut. The following assumes that we always have r_12 < r_cut.
    M12[nfw] = 4*np.pi*rhos[nfw]*rs[nfw]**3*(np.log(1 + x12[nfw]) - x12[nfw]/(1+x12[nfw]))/(h*(1.0 + rs[nfw]/r_cut[nfw])**0.3)
    
    # Otherwise we are in either the core expansion or collapse phase
    easy_core = np.logical_and(np.logical_not(nfw),Halo_r12 < rcore*rs/h)
    # Mass within the half-light radius ; here we take tidal effects via the
    # gravothermal speed up due the change in vmax and rmax, and via the transfer
    # function from Errani & Navarro (2021)
    M12[easy_core] = (4.0/3.0)*np.pi*h**2*rhos[easy_core]*Halo_r12[easy_core]**3 \
                        *rho_core(t_over_t0[easy_core]*vfact[easy_core]**3/rfact[easy_core]**2) \
                            /(1.0 + rs[easy_core]/r_cut[easy_core])**0.3
                            
    # Hardest case: when the half-light radius is larger than the core size, then
    # we have to integrate the profile
    hard_core = np.logical_and(np.logical_not(nfw),Halo_r12 > rcore*rs/h)
    # Place holder for the actual integration of the density profile
    M12[hard_core] = 4.0*np.pi*rhos[hard_core]*rs[hard_core]**3 \
                        *M_half_dimless(t_over_t0[hard_core]*vfact[hard_core]**3/rfact[hard_core]**2,x12[hard_core]) \
                            /(h*(1.0 + rs[hard_core]/r_cut[hard_core])**0.3)

    # Compute sigma_* using the Wolf et al. (2010) approach (arXiv:0908.2995)
    #sigmastar[core] = 1.06114036*(Halo_r12[core]/5e-2)*np.sqrt(h**2*rhos[core]*0.4*f_for_core_density(t_over_t0[core])/1e8)
    sigmastar = np.sqrt(1e-3*M12/(930*Halo_r12))

    if full_out:
        return 10**log_M_peak, vpeak, c_peak, rvir, x12, 1000*Halo_r12, Mr, zinfall, r_peri, r_apoc, t_infall, Torbinfall, num_total_orbits, sigma_o_m, t_over_t0, rcore, 10**log10_T_maxbyT_peri, 10**log10_f_bound, rfact, vfact, r_cut, Prob_s, sigmastar, nfw, easy_core, hard_core, Prob_s_1
    else:
        return  Prob_s, sigmastar, Mr


###############################################################################
# Main likelihood
###############################################################################

def like_sat(in_vec, dat, MV_edge=np.array([-16,-12,-8,-6,-4,-2,0]), sig_edge=np.array([0,2,4,6,8,10,20]),                             mulim=30., Mrlim=-1., alpha=1.87, beta=1.16, K0=1.88e-3):
    """
    Main likelihood which takes a vector of (m_WDM/sigma/m/v_0) and return the likelihood
    of that model.
    """
    m_WDM = in_vec[0]
    sigma_0_o_m_c = 10**in_vec[1]
    v_SIDM_c = 10**in_vec[2]
    M_min = 10**in_vec[3]
    alpha_m = in_vec[4]
    sigma_m = in_vec[5]
    sigma_r = in_vec[6]
    A = in_vec[7]
    n = in_vec[8]
    M_host = in_vec[9]*1.0e12
    
    # Sort boundaries # Check and verify the bound
    if m_WDM < 1.0 or m_WDM > 30.0:
        return -np.inf
    if sigma_0_o_m_c < 1.0e-4 or sigma_0_o_m_c > 200.0:
        return -np.inf
    if v_SIDM_c < 10.0 or v_SIDM_c > 100.0: 
        return - np.inf
    if in_vec[3] < 7.0 or in_vec[3] > 10.0:
        return -np.inf
    if np.arctan(alpha_m) < -1.10715 or np.arctan(alpha_m) > -0.87606:
        return -np.inf
    if sigma_m < 0.0001 or sigma_m > 2.0:
        return -np.inf
    if sigma_r < 0.001 or sigma_r > 2.0:
        return -np.inf
    if A < 1.0 or A > 100.0:
        return -np.inf
    if n < 0.1 or n > 4.0:
        return -np.inf
    if in_vec[9] < 0.5 or in_vec[9] > 3.0:
        return -np.inf
    
    # Generate subhalo population
    num_sims = 50
    num_sims_f = np.float64(num_sims)
    Binned_N = np.zeros((num_sims,MV_edge.size-1,sig_edge.size-1))
    #print(m_WDM, sigma_0_o_m_c, v_SIDM_c, M_min, mulim, Mrlim, alpha, beta, K0, M_host, sigma_r, alpha_m, sigma_m, A, n)
    for i in range(num_sims):
        P, sig, Mr = gen_subhalo_pop(m_WDM, sigma_0_o_m_c, v_SIDM_c, M_min, mulim, Mrlim, alpha, beta, K0, M_host, sigma_r,                            alpha_m, sigma_m, 10**-3*A, n)
        Binned_N[i], xedge, yedge = np.histogram2d(Mr,sig,bins=[MV_edge,sig_edge],weights=P)
            
    sum_term = np.sum(Binned_N, axis=0)

    first_term = ((num_sims_f +1.0)/num_sims_f)**(-(1.0 + sum_term))
    second_term = (num_sims_f +1.0)**(-dat)

    lngam1 = gammaln(1.+ sum_term + dat)
    lngam2 = gammaln(dat + 1.)
    lngam3 = gammaln(1.+ sum_term)
    gam_term = np.exp(lngam1 - lngam2 - lngam3)

    like_array = first_term * second_term * gam_term

    like = np.prod(like_array)

    # Add alpha_m prior
    like *= (1.0/0.21099)*(1.0/(1.0 + alpha_m**2))

    #Add MW mass prior
    #like *= skewnorm.pdf(in_vec[5], 0.2, loc=1.101, scale=0.184)
    like *= norm.pdf(in_vec[9], 1.17, 0.2) #Reference ??

    # Add m_WDM prior
    like *= (1.0/m_WDM)*(1/np.log(30.0/1.0))
    
    return np.log(like)
    like = np.prod(like_array)

    # Add alpha_m prior
    like *= (1.0/0.21099)*(1.0/(1.0 + alpha_m**2))

    #Add MW mass prior
    #like *= skewnorm.pdf(in_vec[5], 0.2, loc=1.101, scale=0.184)
    like *= norm.pdf(in_vec[9], 1.17, 0.2) #Reference ??

    # Add m_WDM prior
    like *= (1.0/m_WDM)*(1/np.log(30.0/1.0))
    
    return np.log(like)