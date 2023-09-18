import numpy as np
import pandas as pd
from astropy.cosmology import Planck18, z_at_value
from scipy import stats
import kalepy as kale
from astropy import constants as const
from astropy import units as u

# Create Data from Simon (2019), Li et. al. (2021).
# Creat infall data from Barmentloo & Cautun (2023), Rocha et. al. (2012), Li et. al. (2017)
# Infall time of Eridanus II is from Li et. al. (2017).
# Infall time of Leo T is fake, it's actually < 1 Gyr.
# Infall time of Pegasus III is fake, same with Pisces II.
# Infall time of Sagittarius is forcefully same with Sagittarius I. 
# Draco II, Hydra II, Segue 2, Triangulum II, Tucana III - Error Bar of Velocity Dispersion has been chosen NaN manually
# Leo T, Pegasus III, Sagittarius - Pericentre and Apocentre Radii have been chosen NaN manually
# Pisces II - Apocentre Radius has been chosen NaN manually; Pericentre (181(+-12) kpc) 
#             has been chosen NaN manually for calculation
# LMC, SMC - Data are from Google and from Drlica-Wagner et. al. (2020), Barmentloo & Cautun (2023), Rocha et. al. (2012)
MW_data = {
    'MW_Sat': ['Aquarius II','Bootes I','Bootes II','Canes Venatici I','Canes Venatici II','Carina','Carina II','Carina III','Coma Berenices','Crater II','Draco','Draco II','Eridanus II','Fornax','Grus I','Hercules','Horologium I','Hydra II','Hydrus I','Large Magellanic Cloud','Leo I','Leo II','Leo IV','Leo V','Leo T','Pegasus III','Pisces II','Reticulum II','Sagittarius','Sculptor','Segue 1','Segue 2','Sextans','Small Magellanic Cloud','Triangulum II','Tucana II','Tucana III','Ursa Major I','Ursa Major II','Ursa Minor','Willman 1'],
    'MW_Sat_Abb': ['AquII','BooI','BooII','CVenI','CVenII','CarI','CarII','CarIII','CberI','CraII','DraI','DraII','EriII','FnxI','GruI','HerI','HorI','HyaII','HyiI','LMC','LeoI','LeoII','LeoIV','LeoV','LeoT','PegIII','PisII','RetII','Sgr','SclI','SegI','SegII','SxtI','SMC','TriII','TucII','TucIII','UMaI','UMaII','UMiI','WilI'],
    'MV': [-4.36,-6.02,-2.94,-8.73,-5.17,-9.45,-4.50,-2.40,-4.28,-8.20,-8.88,-0.80,-7.10,-13.34,-3.47,-5.83,-3.76,-4.86,-4.71,-18.12,-11.78,-9.74,-4.99,-4.29,-8.00,-4.10,-4.23,-3.99,-13.50,-10.82,-1.30,-1.98,-8.94,-17.18,-1.60,-3.90,-1.49,-5.13,-4.43,-9.03,-2.90],
    'MV_err_up': [0.14,0.25,0.74,0.06,0.32,0.05,0.10,0.20,0.25,0.10,0.05,0.40,0.30,0.14,0.59,0.17,0.56,0.37,0.08,np.nan,0.28,0.04,0.26,0.36,np.nan,0.50,0.38,0.38,0.15,0.14,0.73,0.88,0.06,np.nan,0.76,0.20,0.20,0.38,0.26,0.05,0.74],
    'MV_err_low': [0.14,0.25,0.75,0.06,0.32,0.05,0.10,0.20,0.25,0.10,0.05,1.00,0.30,0.14,0.59,0.17,0.56,0.37,0.08,np.nan,0.28,0.04,0.26,0.36,np.nan,0.50,0.38,0.38,0.15,0.14,0.73,0.88,0.06,np.nan,0.76,0.20,0.20,0.38,0.26,0.05,0.74],
    'sigma_star (kms^-1)': [5.4,4.6,10.5,7.6,4.6,6.6,3.4,5.6,4.6,2.7,9.1,5.9,6.9,11.7,2.9,5.1,4.9,3.6,2.7,30.0,9.2,7.4,3.3,2.3,7.5,5.4,5.4,3.3,9.6,9.2,3.7,2.2,7.9,30.0,3.4,8.6,1.2,7.0,5.6,9.5,4.0],
    'sigma_star_err_up (kms^-1)': [3.4,0.8,7.4,0.4,1.0,1.2,1.2,4.3,0.8,0.3,1.2,np.nan,1.2,0.9,2.1,0.9,2.8,np.nan,0.5,np.nan,0.4,0.4,1.7,3.2,1.6,3.0,3.6,0.7,0.4,1.1,1.4,np.nan,1.3,np.nan,np.nan,4.4,np.nan,1.0,1.4,1.2,0.8],
    'sigma_star_err_low (kms^-1)': [0.9,0.6,7.4,0.4,1.0,1.2,0.8,2.1,0.8,0.3,1.2,np.nan,0.9,0.9,1.0,0.9,0.9,np.nan,0.4,np.nan,0.4,0.4,1.7,1.6,1.6,2.5,2.4,0.7,0.4,1.1,1.1,np.nan,1.3,np.nan,np.nan,2.7,np.nan,1.0,1.4,1.2,0.8],
    'r12 (pc)': [160.0,191.0,39.0,437.0,71.0,311.0,92.0,30.0,69.0,1066.0,231.0,19.0,246.0,792.0,28.0,216.0,40.0,67.0,53.0,4735.0,270.0,171.0,114.0,49.0,118.0,78.0,60.0,51.0,2662.0,279.0,24.0,40.0,456.0,2728.0,16.0,121.0,37.0,295.0,139.0,405.0,33.0],
    'r12_err_up (pc)': [26.0,8.0,5.0,18.0,11.0,15.0,8.0,8.0,5.0,86.0,17.0,4.0,17.0,18.0,23.0,20.0,10.0,13.0,4.0,np.nan,17.0,10.0,13.0,16.0,11.0,31.0,10.0,3.0,193.0,16.0,4.0,4.0,15.0,np.nan,4.0,35.0,9.0,28.0,9.0,21.0,8.0],
    'r12_err_low (pc)': [26.0,8.0,5.0,18.0,11.0,15.0,8.0,8.0,4.0,86.0,17.0,3.0,17.0,18.0,23.0,20.0,9.0,13.0,4.0,np.nan,16.0,10.0,13.0,16.0,11.0,25.0,10.0,3.0,193.0,16.0,4.0,4.0,15.0,np.nan,4.0,35.0,9.0,28.0,9.0,21.0,8.0],
    'dist (kpc)': [107.9,66.0,42.0,211.0,160.0,106.0,36.2,27.8,42.0,117.5,82.0,21.5,366.0,139.0,120.0,132.0,87.0,151.0,27.6,50.0,254.0,233.0,154.0,169.0,409.0,205.0,183.0,31.6,26.7,86.0,23.0,37.0,95.0,61.0,28.4,58.0,25.0,97.3,34.7,76.0,45.0],
    'dist_err_up (kpc)': [3.3,2.0,1.0,6.0,4.0,5.0,0.6,0.6,1.6,1.1,6.0,0.4,17.0,3.0,12.0,6.0,13.0,8.0,0.5,3.0,16.0,14.0,5.0,4.0,29.0,20.0,15.0,1.5,1.3,5.0,2.0,3.0,3.0,4.0,1.6,8.0,2.0,6.0,2.0,4.0,10.0],
    'dist_err_low (kpc)': [3.3,2.0,1.0,6.0,4.0,5.0,0.6,0.6,1.5,1.1,6.0,0.4,17.0,3.0,11.0,6.0,11.0,7.0,0.5,3.0,15.0,14.0,5.0,4.0,27.0,20.0,15.0,1.4,1.3,5.0,2.0,3.0,3.0,4.0,1.6,8.0,2.0,5.7,1.9,4.0,10.0],
    'r_peri (kpc)': [101.0,49.0,39.0,62.0,31.0,108.0,27.0,29.0,42.0,33.0,45.0,20.0,345.0,69.0,22.0,52.0,87.0,128.0,25.0,np.nan,56.0,79.0,84.0,164.0,np.nan,np.nan,np.nan,28.0,np.nan,54.0,21.0,19.0,84.0,np.nan,12.0,39.0,3.0,53.0,39.0,42.0,35.0],
    'r_peri_err_up (kpc)': [4.0,5.0,1.0,59.0,48.0,5.0,1.0,1.0,1.0,8.0,6.0,1.0,23.0,16.0,15.0,17.0,13.0,14.0,0.0,np.nan,30.0,72.0,72.0,6.0,np.nan,np.nan,np.nan,2.0,np.nan,3.0,4.0,6.0,4.0,np.nan,1.0,13.0,1.0,33.0,2.0,3.0,31.0],
    'r_peri_err_low (kpc)': [71.0,6.0,1.0,39.0,23.0,17.0,1.0,0.0,1.0,8.0,5.0,1.0,237.0,13.0,13.0,15.0,17.0,71.0,0.0,np.nan,27.0,47.0,69.0,101.0,np.nan,np.nan,np.nan,3.0,np.nan,3.0,5.0,4.0,4.0,np.nan,1.0,11.0,0.0,20.0,2.0,3.0,14.0],
    'r_apoc (kpc)': [116.0,97.0,181.0,242.0,193.0,117.0,252.0,227.0,83.0,133.0,108.0,98.0,485.0,147.0,224.0,212.0,156.0,254.0,167.0,np.nan,888.0,242.0,157.0,197.0,np.nan,np.nan,np.nan,63.0,np.nan,107.0,58.0,47.0,188.0,np.nan,110.0,191.0,47.0,102.0,110.0,92.0,54.0],
    'r_apoc_err_up (kpc)': [190.0,15.0,104.0,26.0,27.0,36.0,34.0,64.0,18.0,2.0,10.0,12.0,822.0,4.0,28.0,37.0,258.0,429.0,29.0,np.nan,578.0,14.0,116.0,460.0,np.nan,np.nan,np.nan,15.0,np.nan,4.0,26.0,4.0,28.0,np.nan,7.0,243.0,6.0,5.0,52.0,4.0,30.0],
    'r_apoc_err_low (kpc)': [8.0,10.0,56.0,8.0,7.0,13.0,26.0,42.0,13.0,2.0,7.0,10.0,50.0,3.0,20.0,20.0,72.0,58.0,23.0,np.nan,185.0,11.0,5.0,18.0,np.nan,np.nan,np.nan,10.0,np.nan,4.0,15.0,3.0,21.0,np.nan,6.0,73.0,4.0,5.0,29.0,3.0,9.0],
    't_infall (Gyrs)': [8.7,8.7,0.7,7.7,8.8,8.5,0.7,0.6,8.2,8.9,9.4,7.4,5.0,8.2,0.4,1.0,8.4,8.4,0.7,0.7,1.7,2.4,8.9,7.5,1.0,0.1,0.1,8.1,8.5,8.7,8.2,7.7,1.0,0.7,8.2,0.7,3.6,9.4,8.2,9.5,8.9],
    't_infall_err_up (Gyrs)': [2.2,2.5,5.9,2.8,1.8,1.8,0.8,0.7,2.9,2.0,2.0,2.2,np.nan,2.2,7.1,5.9,2.7,2.7,5.9,5.6,0.7,5.4,1.5,3.4,np.nan,np.nan,8.1,2.4,1.4,2.1,2.1,2.5,7.0,7.7,1.8,6.6,4.3,1.8,2.9,1.7,1.3],
    't_infall_err_low (Gyrs)': [3.2,2.2,0.7,2.1,2.4,3.4,0.7,0.6,3.6,2.2,2.4,3.8,np.nan,2.7,0.4,1.0,2.8,3.6,0.7,0.7,0.7,1.4,2.9,2.7,np.nan,np.nan,0.1,4.2,4.3,2.2,3.8,3.4,1.0,0.7,3.9,0.7,1.4,2.4,3.9,2.5,4.5],    

}

# Create DataFrame
MW_Satellite_data = pd.DataFrame(MW_data, columns = ['MW_Sat', 'MV', 'MV_err_up', 'MV_err_low', 'sigma_star (kms^-1)', 'sigma_star_err_up (kms^-1)', 'sigma_star_err_low (kms^-1)', 'r12 (pc)', 'r12_err_up (pc)', 'r12_err_low (pc)', 'dist (kpc)', 'dist_err_up (kpc)', 'dist_err_low (kpc)', 'r_peri (kpc)', 'r_peri_err_up (kpc)', 'r_peri_err_low (kpc)', 'r_apoc (kpc)', 'r_apoc_err_up (kpc)', 'r_apoc_err_low (kpc)', 't_infall (Gyrs)', 't_infall_err_up (Gyrs)', 't_infall_err_low (Gyrs)'])

#MW_Satellite_data.to_csv('../mock_data/Milky_Way_Satellites_Observational_Data', sep = ' ', header = 'MW_Sat MV MV_err_up MV_err_low sigma_star (kms^-1) sigma_star_err_up (kms^-1) sigma_star_err_low (kms^-1) r12 (pc) r12_err_up (pc) r12_err_low (pc) dist (kpc) dist_err_up (kpc) dist_err_low (kpc) r_peri (kpc) r_peri_err_up (kpc) r_peri_err_low (kpc) r_apoc (kpc) r_apoc_err_up (kpc) r_apoc_err_low (kpc) t_infall (Gyrs) t_infall_err_up (Gyrs) t_infall_err_low (Gyrs)')



# Infall times of MW Satellites
t_infall_real_data = MW_Satellite_data['t_infall (Gyrs)'].values # units in Gyrs

# Infall redshifts of MW Satellites
z_infall_real_data = np.array(z_at_value(Planck18.lookback_time, t_infall_real_data * u.Gyr, method = 'Bounded'))

# Gaussian KDE
kernel_z_infall = kale.KDE(z_infall_real_data, reflect = True)

# Numpy Array of Radii at Pericentre
Rperi = MW_Satellite_data['r_peri (kpc)'].values
# Numpy Array of Radii at Pericentre without Nan Values
R_peri_real_data = Rperi[np.logical_not(np.isnan(Rperi))] # units in kpc
#R_peri = Rperi[np.logical_and(np.logical_and(np.logical_not(np.isnan(Rperi)), Rperi != 56.0), Rperi != 345.0)]
Rperi_err_low = MW_Satellite_data['r_peri_err_low (kpc)'].values
R_peri_real_data_err_low = Rperi_err_low[np.logical_not(np.isnan(Rperi_err_low))] # units in kpc
Rperi_err_up = MW_Satellite_data['r_peri_err_up (kpc)'].values
R_peri_real_data_err_up = Rperi_err_up[np.logical_not(np.isnan(Rperi_err_up))] # units in kpc

# Numpy Array of Radii at Apocentre
Rapoc = MW_Satellite_data['r_apoc (kpc)'].values
# Numpy Array of Radii at Apocentre without Nan Values
R_apoc_real_data = Rapoc[np.logical_not(np.isnan(Rapoc))] # units in kpc
#R_apoc = Rapoc[np.logical_and(np.logical_not(np.isnan(Rapoc)),Rapoc < 400.0)]
Rapoc_err_low = MW_Satellite_data['r_apoc_err_low (kpc)'].values
R_apoc_real_data_err_low = Rapoc_err_low[np.logical_not(np.isnan(Rapoc_err_low))] # units in kpc
Rapoc_err_up = MW_Satellite_data['r_apoc_err_up (kpc)'].values
R_apoc_real_data_err_up = Rapoc_err_up[np.logical_not(np.isnan(Rapoc_err_up))] # units in kpc

# Numpy Array of Ratio of Peri and Apoc Radii
R_peribyR_apoc_ratio = R_peri_real_data/R_apoc_real_data

# Gaussian KDE
joint_radii = np.vstack([R_apoc_real_data, R_peribyR_apoc_ratio])
radii_KDE = kale.KDE(joint_radii, reflect = True)

# Half-Light Radii of MW Satellites
Halo_r12_real_data = 10**-3 * MW_Satellite_data['r12 (pc)'].values # units in kpc
Halo_r12_real_data_err_low = 10**-3 * MW_Satellite_data['r12_err_low (pc)'].values # units in kpc
Halo_r12_real_data_err_up = 10**-3 * MW_Satellite_data['r12_err_up (pc)'].values # units in kpc


# Absolute Magnitude of MW Satellites
Mr_real_data = MW_Satellite_data['MV'].values
Mr_real_data_err_low = MW_Satellite_data['MV_err_low'].values
Mr_real_data_err_up = MW_Satellite_data['MV_err_up'].values

# Velocity Dispersion of MW Satellites
sigma_star_real_data = MW_Satellite_data['sigma_star (kms^-1)'].values # units in km/s
sigma_star_real_data_err_low = MW_Satellite_data['sigma_star_err_low (kms^-1)'].values # units in km/s
sigma_star_real_data_err_up = MW_Satellite_data['sigma_star_err_up (kms^-1)'].values # units in km/s




