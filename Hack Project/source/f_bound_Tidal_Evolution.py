import numpy as np
import pandas as pd
from scipy import interpolate
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw, concentration


# Load the 1-D Data of Mmx/Mmx0 vs. t/Torb for fixed log10(Tmx/Tperi) = -0.13 from Errani et al (2023)
Mmx_NFW_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data_NFW_rc0.csv')
Mmx_rc100_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-2_rc_1_100.csv')
Mmx_rc30_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-2_rc_1_30.csv')
Mmx_rc10_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-2_rc_1_10.csv')
Mmx_rc3_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-2_rc_1_3.csv')

# 1-D Interpolation
f_Mmx_NFW_1D = interpolate.interp1d(Mmx_NFW_data['x'], Mmx_NFW_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc100_1D = interpolate.interp1d(Mmx_rc100_data['x'], Mmx_rc100_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc30_1D = interpolate.interp1d(Mmx_rc30_data['x'], Mmx_rc30_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc10_1D = interpolate.interp1d(Mmx_rc10_data['x'], Mmx_rc10_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc3_1D = interpolate.interp1d(Mmx_rc3_data['x'], Mmx_rc3_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)

# 2-D Interpolation of Mmx/Mmx0 for fixed log10(Tmx/Tperi) = -0.13

# New data
tbyTorb_Mmx_2D = np.arange(0.0001, 30, 2)
r_coredbyr_s_Mmx = np.array([0.0, 1./100, 1./30, 1./10, 1./3])

# Bound Remnant Function
MmxbyMmx0_function = np.array([f_Mmx_NFW_1D(tbyTorb_Mmx_2D), f_Mmx_rc100_1D(tbyTorb_Mmx_2D), f_Mmx_rc30_1D(tbyTorb_Mmx_2D), f_Mmx_rc10_1D(tbyTorb_Mmx_2D), f_Mmx_rc3_1D(tbyTorb_Mmx_2D)])

#Interpolation
MmxbyMmx0_2D_interp_Tmx_fixed = interpolate.RegularGridInterpolator((r_coredbyr_s_Mmx, tbyTorb_Mmx_2D), MmxbyMmx0_function, method = 'cubic', bounds_error = False, fill_value = -2.5)


# Load the 1-D Data of Mmx/Mmx0 vs. t/Torb for fixed rc/rs = 1/3 from Errani et al (2023)
Mmx_rc3_Tmx_1_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-3_rc_1_3_Tmx_-0.52_1.csv')
Mmx_rc3_Tmx_2_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-3_rc_1_3_Tmx_-0.44_2.csv')
Mmx_rc3_Tmx_3_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-3_rc_1_3_Tmx_-0.36_3.csv')
Mmx_rc3_Tmx_4_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-3_rc_1_3_Tmx_-0.28_4.csv')
Mmx_rc3_Tmx_5_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-3_rc_1_3_Tmx_-0.20_5.csv')
Mmx_rc3_Tmx_6_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-3_rc_1_3_Tmx_-0.13_6.csv')
Mmx_rc3_Tmx_7_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-3_rc_1_3_Tmx_-0.05_7.csv')
Mmx_rc3_Tmx_8_data = pd.read_csv('../Tidal_Evolution_Data/CSV File_Bound Remnant_Errani/Mmx_plot-data-3_rc_1_3_Tmx_0.03_8.csv')

# 1-D Interpolation
f_Mmx_rc3_Tmx_1_data_1D = interpolate.interp1d(Mmx_rc3_Tmx_1_data['x'], Mmx_rc3_Tmx_1_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc3_Tmx_2_data_1D = interpolate.interp1d(Mmx_rc3_Tmx_2_data['x'], Mmx_rc3_Tmx_2_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc3_Tmx_3_data_1D = interpolate.interp1d(Mmx_rc3_Tmx_3_data['x'], Mmx_rc3_Tmx_3_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc3_Tmx_4_data_1D = interpolate.interp1d(Mmx_rc3_Tmx_4_data['x'], Mmx_rc3_Tmx_4_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc3_Tmx_5_data_1D = interpolate.interp1d(Mmx_rc3_Tmx_5_data['x'], Mmx_rc3_Tmx_5_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc3_Tmx_6_data_1D = interpolate.interp1d(Mmx_rc3_Tmx_6_data['x'], Mmx_rc3_Tmx_6_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc3_Tmx_7_data_1D = interpolate.interp1d(Mmx_rc3_Tmx_7_data['x'], Mmx_rc3_Tmx_7_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)
f_Mmx_rc3_Tmx_8_data_1D = interpolate.interp1d(Mmx_rc3_Tmx_8_data['x'], Mmx_rc3_Tmx_8_data[' y'], kind = 'cubic', bounds_error = False, fill_value = -2.5)

# 2-D Interpolation of Mmx/Mmx0 for fixed rc/rs = 1/3

# New data
tbyTorb_Mmx_rc3_Tmx_2D = np.arange(0.0001, 30, 2)
log10_TmxbyTperi_rc3_2D = np.array([-0.52, -0.44, -0.36, -0.28, -0.20, -0.13, -0.05, 0.03])

# Density Contrast Function
Mmx_rc3_Tmx_function = np.array([f_Mmx_rc3_Tmx_1_data_1D(tbyTorb_Mmx_rc3_Tmx_2D), f_Mmx_rc3_Tmx_2_data_1D(tbyTorb_Mmx_rc3_Tmx_2D), f_Mmx_rc3_Tmx_3_data_1D(tbyTorb_Mmx_rc3_Tmx_2D), f_Mmx_rc3_Tmx_4_data_1D(tbyTorb_Mmx_rc3_Tmx_2D), f_Mmx_rc3_Tmx_5_data_1D(tbyTorb_Mmx_rc3_Tmx_2D), f_Mmx_rc3_Tmx_6_data_1D(tbyTorb_Mmx_rc3_Tmx_2D), f_Mmx_rc3_Tmx_7_data_1D(tbyTorb_Mmx_rc3_Tmx_2D), f_Mmx_rc3_Tmx_8_data_1D(tbyTorb_Mmx_rc3_Tmx_2D)])

#Interpolation
MmxbyMmx0_2D_interp_rcore_fixed = interpolate.RegularGridInterpolator((log10_TmxbyTperi_rc3_2D, tbyTorb_Mmx_rc3_Tmx_2D), Mmx_rc3_Tmx_function, method = 'cubic', bounds_error = False, fill_value = -2.5)

# 1-D Interpolation for fixed rc/rs = 1/3 and log10(Tmx/Tperi) = -0.13
MmxbyMmx0_1D_interp_rcore_Tmx_fixed = f_Mmx_rc3_Tmx_6_data_1D

