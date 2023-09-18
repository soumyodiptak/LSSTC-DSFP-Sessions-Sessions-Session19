#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 11:39:46 2019

@author: Francis-Yan Cyr-Racine, University of New Mexico

Code to make and validate mock satellite data

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import Milky_Way_Satellites_WDM_SIDM_Model
from MWS_real_data_distribution import Mr_real_data, Mr_real_data_err_low, Mr_real_data_err_up, Halo_r12_real_data, Halo_r12_real_data_err_low, Halo_r12_real_data_err_up, sigma_star_real_data, sigma_star_real_data_err_low, sigma_star_real_data_err_up

# Generic properties from matplotlib
rcParams['font.family'] = 'serif'
rcParams['text.usetex'] = True

# Datafile name
data_dir = '../mock_data/'
filename = 'Milky_Way_Satellites_WDM_SIDM_Mock_Data'

# Set parameters
m_WDM = 6.0 # keV
sigma_0_o_m_c = 300.0 # cm^2/g, The normalization factor in the velocity dependent
                      # dark matter self-interaction cross section over mass
v_SIDM_c = 50.0 # km/s, The velocity scale above which the sigma_0_o_m_c falls off as v^-4, in units of km/s.
log_M_min = 7.5 # threshold for star formation
alpha=1.87 # Slope of subhalo peak mass function
beta=1.16 # shape parameter for WDM mass function
K0=1.88e-3 # Normalization of the subhalo mass function
M_host = 1.17e12 # MW mass
alpha_m = -1.31 # faint-end slope of the luminosity functioin
sigma_m = 0.005 # scatter in luminosity function
sigma_r = 0.20 # scatter of half-light rad relation
A = 20.0 # Normalization of the half light-radius relation (in pc) #Likelihood 
n = 1.5 # slope of the half-light radius relation #Likelihood
sigma_c_0 = 0.1 # Amplitude of the scatter of the mass-concentration relation, normalized to M = 1e9M_sun #Likelihood
zeta = -0.33 # Slope of the scatter of the mass-concentration relation #Likelihood

# Observation parameters
mulim = 30 # Surface brightness limit
Mrlim = -1.0 # Absolute magnitude limit

# Make dataset
M_peak, v_peak, c_peak, r_vir, x12, Halo_r12, Mr, z_infall, r_peri, r_apoc, t_infall, Torb_infall, num_total_orbits, sigma_o_m, t_over_t0, r_core, T_maxbyT_peri, f_bound, r_fact, v_fact, r_cut, Prob_surv, sigma_star, nfw, easy_core, hard_core, Prob_s_ML = Milky_Way_Satellites_WDM_SIDM_Model.gen_subhalo_pop(20.0,0.0001,0.0001,M_min=10**log_M_min, alpha=alpha,beta=beta,K0=K0, alpha_m=alpha_m,sigma_m=sigma_m, mulim=mulim,Mrlim=Mrlim, M_host=M_host,sigma_r=sigma_r, A=10**-3*A,n=n,sigma_c_0=sigma_c_0, zeta=zeta,full_out=True)


M_peak_sim, v_peak_sim, c_peak_sim, r_vir_sim, x12_sim, Halo_r12_sim, Mr_sim, z_infall_sim, r_peri_sim, r_apoc_sim, t_infall_sim, Torb_infall_sim, num_total_orbits_sim, sigma_o_m_sim, t_over_t0_sim, r_core_sim, T_maxbyT_peri_sim, f_bound_sim, r_fact_sim, v_fact_sim, r_cut_sim, Prob_surv_sim, sigma_star_sim, nfw_sim, easy_core_sim, hard_core_sim, Prob_s_ML_sim  = Milky_Way_Satellites_WDM_SIDM_Model.gen_subhalo_pop(m_WDM,sigma_0_o_m_c,v_SIDM_c,M_min=10**log_M_min, alpha=alpha,beta=beta,K0=K0, alpha_m=alpha_m,sigma_m=sigma_m, mulim=mulim,Mrlim=Mrlim, M_host=M_host,sigma_r=sigma_r, A=10**-3*A,n=n,sigma_c_0=sigma_c_0, zeta=zeta, full_out=True)



fig, ax = plt.subplots(1,2,figsize=(16, 6))
ax1 = ax[0]
ax1.scatter(Mr[nfw],Halo_r12[nfw],s=1,c='b',label=r'CDM NFW')
ax1.scatter(Mr[easy_core],Halo_r12[easy_core],s=4,c='g',label=r'CDM Easy core')
ax1.scatter(Mr[hard_core],Halo_r12[hard_core],s=4,c='b',label=r'CDM Hard core')
ax1.scatter(Mr_sim[nfw_sim],Halo_r12_sim[nfw_sim],s=6,c='r',marker='s',label=r'SIDM NFW')
ax1.scatter(Mr_sim[easy_core_sim],Halo_r12_sim[easy_core_sim],s=6,c='m',marker='s',label=r'SIDM easy core')
ax1.scatter(Mr_sim[hard_core_sim],Halo_r12_sim[hard_core_sim],s=6,c='k',marker='s',label=r'SIDM hard core')
ax1.errorbar(Mr_real_data,1000*Halo_r12_real_data,xerr=[Mr_real_data_err_low, Mr_real_data_err_up],yerr=[1000*Halo_r12_real_data_err_low, 1000*Halo_r12_real_data_err_up],marker='s',ls='',c='orange',alpha=0.7,ms=4,label=r'MW Satellites')
#ax1.scatter(Mr,r12,s=1,c='b',label=r'CDM')
#ax1.scatter(Mr_sim,r12_sim,s=3,c='r',label=r'WDM')
ax1.set_xlabel(r'$M_V$',fontsize=20)
ax1.set_ylabel(r'$r_{1/2}$ [pc]',fontsize=20)
ax1.set_xlim(0,-20)
#ax1.set_ylim(0,13)
#ax1.set_xscale('log')
ax1.set_yscale('log')
ticklabels = ax1.get_xticklabels()
ticklabels.extend(ax1.get_yticklabels())
for label in ticklabels:
    label.set_color('k')
    label.set_fontsize(20)
ax1.legend(fontsize=14)

ax1 = ax[1]
ax1.scatter(Mr[nfw],sigma_star[nfw],s=1,c='b',label=r'CDM NFW')
ax1.scatter(Mr[easy_core],sigma_star[easy_core],s=4,c='g',label=r'CDM easy core')
ax1.scatter(Mr[hard_core],sigma_star[hard_core],s=4,c='b',label=r'CDM hard core')
ax1.scatter(Mr_sim[nfw_sim],sigma_star_sim[nfw_sim],s=6,c='r',marker='s',label=r'SIDM NFW')
ax1.scatter(Mr_sim[easy_core_sim],sigma_star_sim[easy_core_sim],s=6,c='m',marker='s',label=r'SIDM easy core')
ax1.scatter(Mr_sim[hard_core_sim],sigma_star_sim[hard_core_sim],s=6,c='k',marker='s',label=r'SIDM hard core')
ax1.errorbar(Mr_real_data,sigma_star_real_data,xerr=[Mr_real_data_err_low, Mr_real_data_err_up],yerr=[sigma_star_real_data_err_low, sigma_star_real_data_err_up],marker='s',ls='',c='orange',alpha=0.7,ms=4,label=r'MW Satellites')
ax1.set_xlabel(r'$M_V$',fontsize=20)
ax1.set_ylabel(r'$\sigma_\star$ [km/s]',fontsize=20)
ax1.set_xlim(0,-20)
ax1.set_ylim(0,35)
#ax1.set_xscale('log')
#ax1.set_yscale('log')
ticklabels = ax1.get_xticklabels()
ticklabels.extend(ax1.get_yticklabels())
for label in ticklabels:
    label.set_color('k')
    label.set_fontsize(20)
ax1.legend(fontsize=14)

# Note Leo T error bar on MV is fake 
#MW_sats_name = ['Sculptor','Hydrus I','Fornax','Horologium I','Reticulum II','Eridanus II','Carina','Carina II','Carina III','Ursa Major II', \
#                'Leo T','Segue I','Leo I','Sextans','Ursa Major I','Willman 1','Leo II','Leo V','Leo IV','Crater II','Coma #Berenices','Canes Venatici II', \
#                'Canes Venatici I','Bootes II','Bootes I','Ursa Minor','Hercules','Draco','Sagittarius','Pegasus III','Aquarius #II','Tucana II','Grus I','Pisces II']
#MV_MW = np.array([-10.82,-4.71,-13.34,-3.76,-3.99,-7.10,-9.45,-4.50,-2.40,-4.43,-8,-1.30,-11.78,-8.94,-5.13,-2.90,-9.74,-4.29,-4.99,-8.2,-4.28,-5.17,-8.73,-2.94,-6.02,-9.03,-5.83,-8.88,-13.5,-4.1,-4.36,-3.90,-3.47,-4.23])
#e_MV_MW =np.array([0.14,0.08,0.14,0.56,0.38,0.3,0.05,0.1,0.2,0.26,0.001,0.73,0.28,0.06,0.38,0.74,0.04,0.36,0.26,0.10,0.25,0.32,0.06,0.75,0.25,0.05,0.17,0.05,0.15,0.5,0.14,0.2,0.59,0.38])

#half-light radius
#R12_MW = np.array([279,53,792,40,51,246,311,92,30,139,118,24,270,456,295,33,171,49,114,1066,69,71,437,39,191,405,216,231,2662,78,160,121,28,60])
#e_R12_MW = np.array([16,4,18,10,3,17,15,8,8,9,11,4,17,15,28,8,10,16,13,86,5,11,18,5,8,21,20,17,193,31,26,35,23,10])


#sigma_star_MW = np.array([9.2,2.7,11.7,4.9,3.3,6.9,6.6,3.4,5.6,5.6,7.5,3.7,9.2,7.9,7.0,4.0,7.4,2.3,3.3,2.7,4.6,4.6,7.6,10.5,4.6,9.5,5.1,9.1,9.6,5.4,5.4,8.6,2.9,5.4])
#sigma_star_MW_eUp = [1.1,0.5,0.9,2.8,0.7,1.2,1.2,1.2,4.3,1.4,1.6,1.4,0.4,1.3,1.0,0.8,0.4,3.2,1.7,0.3,0.8,1.0,0.4,7.4,0.8,1.2,0.9,1.2,0.4,3.0,3.4,4.4,2.1,3.6]
#sigma_star_MW_eLow = [1.1,0.4,0.9,0.9,0.7,0.9,1.2,0.8,2.1,1.4,1.6,1.1,0.4,1.3,1.0,0.8,0.4,1.6,1.7,0.3,0.8,1.0,0.4,7.4,0.6,1.2,0.9,1.2,0.4,2.5,0.9,2.7,1.0,2.4]
#sigma_star_MW_error = np.array([sigma_star_MW_eLow,sigma_star_MW_eUp])

# 
#MW_sats_name_justupper = ['Triangulum II','Segue 2','Hydra II','Tucana III','Draco II']
#R12_MW_justupper = np.array([16,40,67,37,19])
#e_R12_MW_justupper = np.array([4,4,13,9,4])
#MW_MV_justupper=np.array([-1.60,-1.98,-4.86,-1.49,-0.80])
#MW_MV_justupper_e=np.array([0.76,0.88,0.37,0.2,1])
#sigma_star_MW_upper = np.array([3.4,2.2,3.6,1.2,5.9])

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.set_xlabel(r'$M_V$',fontsize=25)
ax2.set_ylabel(r'$\sigma_\star$ [km/s]',fontsize=25)
ax2.scatter(Mr,sigma_star,s=9,c='g',label=r'CDM')
ax2.scatter(Mr_sim,sigma_star_sim,s=9,c='r',marker='s',label=r'$m_{\rm WDM} = $' + str(m_WDM) + r' keV, $\frac{\sigma_{\rm 0}}{m} = $' + str(sigma_0_o_m_c) + r' cm$^2$/g, $v_{\rm 0} = $' + str(v_SIDM_c) + r' km/s')
#ax2.errorbar(MV_MW,sigma_star_MW,yerr=sigma_star_MW_error ,xerr=e_MV_MW,marker='s',ls='',c='orange',alpha=0.7,ms=4,label=r'MW Satellites')
#ax2.errorbar(MW_MV_justupper,sigma_star_MW_upper ,yerr=sigma_star_MW_upper ,xerr=MW_MV_justupper_e,marker='s',ls='',ms=4,uplims=True,c='orange',alpha=0.7)
ax2.errorbar(Mr_real_data,sigma_star_real_data,xerr=[Mr_real_data_err_low, Mr_real_data_err_up],yerr=[sigma_star_real_data_err_low, sigma_star_real_data_err_up],marker='s',ls='',c='orange',alpha=0.7,ms=4,label=r'MW Satellites')
ax2.invert_xaxis()
ax2.set_xlim(0,-16)
ax2.set_ylim(0,20)
#ax2.set_xscale('log')
#ax2.set_yscale('log')
ticklabels = ax2.get_xticklabels()
ticklabels.extend(ax2.get_yticklabels())
for label in ticklabels:
    label.set_color('k')
    label.set_fontsize(20)
ax2.legend(fontsize=10)
plt.show()
#fig2.savefig('../plots/Disp_vs_lum_MW_sats_wdm_sidm_singlesort.pdf',bbox_inches='tight')
fig2.savefig('../plots/Velocity_Dispersion_vs_Luminosity_Milky_Way_Satellites_WDM_SIDM_Mock_Data.pdf',bbox_inches='tight')

print('CDM:', np.sum(Prob_surv), 'WDM:',np.sum(Prob_surv_sim))

# Save the data
test_h = np.vstack((M_peak_sim, v_peak_sim, c_peak_sim, r_vir_sim, Halo_r12_sim, Prob_surv_sim, sigma_star_sim, Mr_sim, nfw_sim, easy_core_sim, hard_core_sim))
#test_h = np.vstack((m_peak,v_peak,c_peak, rvir_peak, r12, ps, sigmastar, Mr, nfw, easy_core, hard_core))
np.savetxt(data_dir + filename + '.dat',test_h.T, header='M_peak (M_sun) v_peak (km/s) c_peak r_vir_peak (kpc) r_12 (pc) Prob_surv sigma_star (km/s) M_V NFW Easy_core Hard_core')

# Save parameters
output_file = data_dir + filename + '_params.dat'
f = open(output_file, 'w')
f.write('m_WDM: {:03.4f} keV\n'.format(m_WDM))
f.write('sigma_0_o_m_c: {:03.4f} cm^2/g\n'.format(sigma_0_o_m_c))
f.write('v_SIDM_c: {:03.4f} km/s\n'.format(v_SIDM_c))
f.write('log_M_min: {:03.4f} \n'.format(log_M_min))
f.write('alpha: {:03.4f}\n'.format(alpha))
f.write('beta: {:03.4f}\n'.format(beta))
f.write('K0: {:03.4e} 1/M_sun \n'.format(K0))
f.write('M_host: {:03.4e} M_sun\n'.format(M_host))
f.write('alpha_m: {:03.4f} \n'.format(alpha_m))
f.write('sigma_m: {:03.4f}\n'.format(sigma_m))
f.write('sigma_r: {:03.4f} \n'.format(sigma_r))
f.write('A: {:03.4f} kpc\n'.format(A))
f.write('n: {:03.4f} \n'.format(n))
f.write('mulim: {:03.2f}\n'.format(mulim))
f.write('Mrlim: {:03.2f} \n'.format(Mrlim))
f.close()
