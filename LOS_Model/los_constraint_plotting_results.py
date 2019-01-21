"""
LOS constraint variation analysis, plotting script for results...

16th Oct. 2018

"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.optimize import curve_fit
import sympy as sym
plt.ion()

"""
#c207?
taxi_los_results_path = '/home/user/Dropbox/RandomDataResults/' 
path_to_figures = '/home/user/Dropbox/vnc/'
"""


#klara
taxi_los_results_path = '/home/pdawg/Dropbox/RandomDataResults/' 
path_to_figures = '/home/pdawg/Pictures/'



sf_los_results_file1 = 'SF_2_days_LOS_contraint_investigation_results.pickle'
rome_los_results_file1 = 'Rome_14_days_LOS_contraint_investigation_results.pickle'

sf_los_results_file2 = 'SF_4_days_LOS_contraint_investigation_results.pickle'
rome_los_results_file2 = 'Rome_21_days_LOS_contraint_investigation_results.pickle'


sf_taxi_los_results_dict = pickle.load( open( (taxi_los_results_path+sf_los_results_file2), 'rb'))
rome_taxi_los_results_dict = pickle.load( open( (taxi_los_results_path+rome_los_results_file2), 'rb'))

def CombinedLOSResultsDicts(taxi_los_results_path ,sf_los_results_file1,rome_los_results_file1,sf_los_results_file2 ,rome_los_results_file2):

    sf_taxi_los_results_dict1 = pickle.load( open( (taxi_los_results_path+sf_los_results_file1), 'rb'))
    rome_taxi_los_results_dict1 = pickle.load( open( (taxi_los_results_path+rome_los_results_file1), 'rb'))

    sf_taxi_los_results_dict2 = pickle.load(open( (taxi_los_results_path+sf_los_results_file2), 'rb'))
    rome_taxi_los_results_dict2 = pickle.load( open( (taxi_los_results_path+rome_los_results_file2), 'rb'))
    
    combined_sf_nolos_results = sf_taxi_los_results_dict2['nolos'] + sf_taxi_los_results_dict1['nolos']
    combined_sf_los_results = sf_taxi_los_results_dict2['los'] + sf_taxi_los_results_dict2['los']

    combined_rome_nolos_results = rome_taxi_los_results_dict2['nolos'] + rome_taxi_los_results_dict1['nolos']
    combined_rome_los_results = rome_taxi_los_results_dict2['los'] + rome_taxi_los_results_dict2['los']
    
    combined_sf_taxi_los_results_dict = {'nolos':combined_sf_nolos_results, 'los':combined_sf_los_results}
    combined_rome_taxi_los_results_dict = {'nolos':combined_rome_nolos_results, 'los':combined_rome_los_results}
    return combined_sf_taxi_los_results_dict, combined_rome_taxi_los_results_dict



combined_sf_taxi_los_results_dict, combined_rome_sf_taxi_results_dict = CombinedLOSResultsDicts(taxi_los_results_path ,sf_los_results_file1,rome_los_results_file1,sf_los_results_file2 ,rome_los_results_file2)

histo_bins = [0,50,100,150,200,250,300,350,400,450,500]




# SF histogram plot...

sf_los_results_histo = np.histogram(sf_taxi_los_results_dict['los'],bins = histo_bins, density=True)
sf_nolos_results_histo = np.histogram(sf_taxi_los_results_dict['nolos'], bins = histo_bins, density=True)

plotting_bins = np.array(histo_bins[1:]) - 25 #np.cumsum(np.diff(sf_los_results_histo[1]))

"""
plt.figure()
plt.plot(plotting_bins,sf_los_results_histo[0],'or',plotting_bins,sf_nolos_results_histo[0],'sg')
plt.xlabel('Distance/[m]')
plt.ylabel('Some form of Frequency...?')
plt.title('SF 2dayz')
plt.show()
"""

#Rome Histogram Plot
rome_los_results_histo = np.histogram(rome_taxi_los_results_dict['los'],bins = histo_bins, density=True)
rome_nolos_results_histo = np.histogram(rome_taxi_los_results_dict['nolos'], bins = histo_bins, density=True)

"""
plt.figure()
plt.plot(plotting_bins,rome_los_results_histo[0],'or',plotting_bins,rome_nolos_results_histo[0],'sg')
plt.xlabel('Distance/[m]')
plt.ylabel('Some form of Frequency...?')
plt.title('Rome 14dayz')
plt.show()

#Combined Rome+SF histo plot
plt.figure(figsize=(8,6))
plt.plot(plotting_bins,rome_los_results_histo[0],'-.og',plotting_bins,rome_nolos_results_histo[0],'-.or', plotting_bins,sf_los_results_histo[0],'--sg',plotting_bins,sf_nolos_results_histo[0],'--sr') 
plt.legend(['Rome-LOS','Rome-NOLOS','SF-LOS','SF-NOLOS'])
plt.xlabel('Distance/[m]')
plt.ylabel('Probability Density Function')
plt.title('Rome and SF')
plt.savefig((path_to_figures+'rome_sf_los_constraint_variation_results_PDF_plot.pdf'),dpi=400) 
plt.show()
"""
# Combined Rome and SF, corrected for 'REAL' frequency...
sf_los_results_histo = np.histogram(sf_taxi_los_results_dict['los'],bins = histo_bins, density=False)
sf_nolos_results_histo = np.histogram(sf_taxi_los_results_dict['nolos'], bins = histo_bins, density=False)
rome_los_results_histo = np.histogram(rome_taxi_los_results_dict['los'],bins = histo_bins, density=False)
rome_nolos_results_histo = np.histogram(rome_taxi_los_results_dict['nolos'], bins = histo_bins, density=False)

freq_rome_los_results = rome_los_results_histo[0] #*len(rome_taxi_los_results_dict['los'])
freq_rome_nolos_results = rome_nolos_results_histo[0] #*len(rome_taxi_los_results_dict['nolos'])

freq_sf_los_results = sf_los_results_histo[0] #*len(sf_taxi_los_results_dict['los'])
freq_sf_nolos_results = sf_nolos_results_histo[0] #*len(sf_taxi_los_results_dict['nolos'])

"""
plt.figure(figsize=(8,6))
plt.plot(plotting_bins,freq_rome_los_results,'-.og',plotting_bins,freq_rome_nolos_results,'-.or', plotting_bins,freq_sf_los_results,'--sg',plotting_bins,freq_sf_nolos_results,'--sr') 
plt.legend(['Rome-LOS','Rome-NOLOS','SF-LOS','SF-NOLOS'])
plt.xlabel('Distance/[m]')
plt.ylabel('Frequency Count')
plt.title('Rome and SF')
plt.savefig((path_to_figures+'rome_sf_los_constraint_variation_results_freq_plot.pdf'),dpi=400) 
plt.show()
"""


# Combined Rome and SF, real frequency, divded through by class (total freq count in each bin)... note, p(nolos) + p(los) = 1 for each bin...

rome_freq_results_ALL = freq_rome_los_results + freq_rome_nolos_results
sf_freq_results_ALL = freq_sf_los_results + freq_sf_nolos_results

"""
plt.figure(figsize=(8,6))
plt.plot(plotting_bins,freq_rome_los_results/rome_freq_results_ALL,'-.og',plotting_bins,freq_rome_nolos_results/rome_freq_results_ALL,'-.or', plotting_bins,freq_sf_los_results/sf_freq_results_ALL,'--sg',plotting_bins,freq_sf_nolos_results/sf_freq_results_ALL,'--sr') 
plt.legend(['Rome-LOS','Rome-NOLOS','SF-LOS','SF-NOLOS'])
plt.xlabel('Distance/[m]')
plt.ylabel('within bin probability')
plt.title('Rome and SF')
plt.savefig((path_to_figures+'rome_sf_los_constraint_variation_results_Within_bin_probability_plot.pdf'),dpi=400) 
plt.show()
"""

# modified figure, for da VNC paper....
plt.figure(figsize=(8*0.75,6*0.75))
plt.plot(plotting_bins,freq_rome_los_results/rome_freq_results_ALL,'-or', plotting_bins,freq_sf_los_results/sf_freq_results_ALL,'-sk') 
plt.legend(['Rome-LOS','San Francisco-LOS'])
plt.xlabel('Euclidean Distance Between Taxi Pairs/[m]')
plt.ylabel("P(Taxi pair have LOS | Taxi pair separation distance)")
plt.xticks(np.arange(0,(max(histo_bins)+50),50))
#plt.title('Rome and SF LOS probability')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
plt.savefig((path_to_figures+'rome_sf_los_constraint_variation_results_Within_bin_probability_plot.pdf'),dpi=400) 
plt.show()

"""
LOS_DATA_DUMP_DICT = {'rome_dist':plotting_bins, 'rome_LOS_prob':freq_rome_los_results/rome_freq_results_ALL, 'SF_dist':plotting_bins, 'SF_LOS_prob':freq_sf_los_results/sf_freq_results_ALL}

with open(('LOS_DATA_DUMP.pickle'), 'wb') as handle3:
    pickle.dump(LOS_DATA_DUMP_DICT, handle3, protocol = pickle.HIGHEST_PROTOCOL)

"""

# now with expo-something curve fitting


x = np.concatenate((plotting_bins,plotting_bins),axis=None)
y_ground_truth = np.concatenate(((freq_rome_los_results/rome_freq_results_ALL),(freq_sf_los_results/sf_freq_results_ALL)),axis=None)

def FitExpoFunc(x,a,b):
    return a*(np.exp(-b*x))

expo_y_fit = FitExpoFunc(x,0.8058,0.004)
#expo_yn = expo_y_fit + 0.1*np.random.normal(size=len(x))
expo_popt, expo_pcov = curve_fit(FitExpoFunc, x, expo_y_fit)
#def FitFunc(x,a,b,c,d):
#    return a*(1/((x+b)**c))+d

def FitPolyFunc(x,a,b,c,d):
    return a*x**3 + b*x**2 + c*x + d

poly_y_fit = FitPolyFunc(x,-2*(10**-8),2*(10**-5),-0.0069,1.02792)
poly_popt, poly_pcov = curve_fit(FitPolyFunc, x, poly_y_fit)





plt.figure(figsize=(8*0.75,6*0.75))
plt.plot(plotting_bins,freq_rome_los_results/rome_freq_results_ALL,'or', plotting_bins,freq_sf_los_results/sf_freq_results_ALL,'sk') 
plt.plot(x, FitPolyFunc(x, *poly_popt), '+b', label='Fitted Curve')
plt.legend(['Rome-LOS','San Francisco-LOS', 'Fitted Curve'])
plt.xlabel('Euclidean Distance Between Taxi Pairs/[m]')
plt.ylabel("P(Taxi pair have LOS | Taxi pair separation distance)")
plt.xticks(np.arange(0,(max(histo_bins)+50),50))
#plt.title('Rome and SF LOS probability')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
plt.savefig((path_to_figures+'rome_sf_los_constraint_variation_results_with_curve_Fit_plot.pdf'),dpi=400) 
plt.show()



x_fitted_curve = np.arange(0,500,2)
y_fitted_poly_curve = FitPolyFunc(x_fitted_curve,-2*(10**-8),2*(10**-5),-0.0069,1.02792)
y_fitted_expo_curve = FitExpoFunc(x_fitted_curve,0.8058,0.004)


xs = sym.Symbol('\lambda')
poly_tex = sym.latex(FitPolyFunc(xs,*poly_popt)).replace('$','')

#fudge_expo_popt = np.array([0.8058,0.004])
#expo_tex = sym.latex(FitExpoFunc(xs,*fudge_expo_popt)).replace('$','')


plt.figure(figsize=(8*0.75,6*0.75))
plt.plot(plotting_bins,freq_rome_los_results/rome_freq_results_ALL,'or', plotting_bins,freq_sf_los_results/sf_freq_results_ALL,'sk') 
plt.plot(x_fitted_curve, y_fitted_poly_curve, '--b', label='Polynomial-Fitted Curve')
plt.plot(x_fitted_curve, y_fitted_expo_curve, '-.b', label='Exponential-Fitted Curve')
#plt.legend(['Rome-LOS','San Francisco-LOS', (r'$f(\lambda)= %s$' %(poly_tex))]) #, (r'$f(\lambda)= %s$' %(expo_tex))])
plt.legend(['Rome-LOS','San Francisco-LOS', (r'$f(\lambda)= %s$' %(poly_tex)), 'Exponential-Fitted Curve']) #, (r'$f(\lambda)= %s$' %(expo_tex))])
plt.xlabel('Euclidean Distance Between Taxi Pairs/[m]')
plt.ylabel("P(Taxi pair have LOS | Taxi pair separation distance)")
plt.xticks(np.arange(0,(max(histo_bins)+50),50))
#plt.title('Rome and SF LOS probability')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
plt.savefig((path_to_figures+'rome_sf_los_constraint_variation_results_with_curve_Fit_plot.pdf'),dpi=400) 
plt.show()



"""
# MEGA Combined! two datasets for each Rome and SF
combined_sf_los_results_histo = np.histogram(combined_sf_taxi_los_results_dict['los'],bins = histo_bins, density=True)
combined_sf_nolos_results_histo = np.histogram(combined_sf_taxi_los_results_dict['nolos'], bins = histo_bins, density=True)
combined_rome_los_results_histo = np.histogram(combined_rome_sf_taxi_results_dict['los'],bins = histo_bins, density=True)
combined_rome_nolos_results_histo = np.histogram(combined_rome_sf_taxi_results_dict['nolos'], bins = histo_bins, density=True)


plt.figure(figsize=(8,6))
plt.plot(plotting_bins,combined_rome_los_results_histo[0],'-.og',plotting_bins,combined_rome_nolos_results_histo[0],'-.or', plotting_bins,combined_sf_los_results_histo[0],'--sg',plotting_bins,combined_sf_nolos_results_histo[0],'--sr') 
plt.legend(['Rome-LOS','Rome-NOLOS','SF-LOS','SF-NOLOS'])
plt.xlabel('Distance/[m]')
plt.ylabel('PDF')
plt.title('Rome and SF, Multiple Data-sets')
plt.savefig((path_to_figures+'multiple_datasets_rome_sf_los_constraint_variation_results_PDF_plot.png'),dpi=400) 
plt.show()
"""





"""
#NOTES section....


https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly







"""
