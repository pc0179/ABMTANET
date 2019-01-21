"""
a hillarious script

takes LOS analysis in urban areas (SF and Roma) and fits random lines to it, hopefully to get some useful distribution out of this...


"""
import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.ion()

"""
#On Klara...
filename = 'LOS_DATA_DUMP.pickle'
datafilepath = '/home/pdawg/MiniTaxiFleets/LineOfSight/'
"""

#on c207 jan15
datafilepath = '/home/user/MiniTaxiFleets/LineOfSight/'
filename = 'LOS_DATA_DUMP.pickle'


LOS_DATA_RESULTS = pickle.load( open( (datafilepath+filename), 'rb'))


p_v2v_sf_los = LOS_DATA_RESULTS['SF_LOS_prob']
p_v2v_rome_los = LOS_DATA_RESULTS['rome_LOS_prob']
v2v_dist = LOS_DATA_RESULTS['rome_dist']

"""
plt.figure()
plt.plot(v2v_dist,p_v2v_sf_los,'-sk',v2v_dist,p_v2v_rome_los,'-or')
plt.xlabel('V2V Euclidean distance')
plt.ylabel('Likelihood of LOS')
plt.title('why bother?')
"""




normalised_distances = v2v_dist/500 #v2v_dist.max()

"""
#note distances go from 25-475m#
dart_throw_dict = dict()
dart_throw_dict[0] = np.exp(-1*normalised_distances)
dart_throw_dict[1] = 0.36666666666667*np.exp(-1*normalised_distances)
dart_throw_dict[2] = 0.014*np.exp(np.sqrt(1/normalised_distances))
dart_throw_dict[3] = 0.03*np.exp(np.sqrt(1/normalised_distances))
dart_throw_dict[4] = 0.03*np.exp(np.sqrt(3/normalised_distances))
dart_throw_dict[5] = 0.03*np.exp(np.sqrt(2/normalised_distances))
dart_throw_dict[6] = 0.03*np.exp(np.sqrt(1.78/normalised_distances))
dart_throw_dict[6] = 0.025*np.exp(np.sqrt(1.9/normalised_distances))




COLOUR_NUM = len(dart_throw_dict.keys())
colours_list=iter(plt.cm.rainbow(np.linspace(0,1,COLOUR_NUM)))





plt.figure()
for key, value in dart_throw_dict.items():

    plotting_colour = next(colours_list)
    plt.plot(normalised_distances, value,c=plotting_colour)
    print(key)
    print(value)

plt.plot(normalised_distances,p_v2v_sf_los,'-sk',normalised_distances,p_v2v_rome_los,'-or')
plt.ylim((0,1))
plt.xlim((0,1))
plt.xlabel('V2V Euclidean distance')
plt.ylabel('Likelihood of LOS')
plt.legend(list(dart_throw_dict.keys()))
plt.title('why bother?')
"""

mean_p_v2v = (p_v2v_rome_los + p_v2v_sf_los)/2
ln_p_v2v = np.log(mean_p_v2v)
lognormal_distances = np.log(normalised_distances)

#polyfit_quiffs = np.polyfit(mean_p_v2v, lognormal_distances,1)

polyfit_quiffs2 = np.polyfit(normalised_distances, np.log(mean_p_v2v),1)


x = np.linspace(0.01,1,100)

polyfit_quiffs = [-0.27373108,0.119545]
polyfit_results = polyfit_quiffs2[0]*np.log(x) + polyfit_quiffs2[1]

plt.figure()
plt.plot(lognormal_distances, mean_p_v2v,'-ok',np.log(x), polyfit_results,'*g')

plt.xlabel('LogNormal V2V Euclidean distance')
plt.ylabel('Likelihood of LOS')
plt.title('You are a fool.')



"""
https://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
polyfit_quiffs = [-0.27373108,0.119545]
"""

x3 = normalised_distances
y3 = mean_p_v2v


polyfit_quiffs4 = np.polyfit(x3,np.log(y3),1, w=np.sqrt(y3))
polyfit_results4 = np.log(polyfit_quiffs4[1]) + polyfit_quiffs4[0]*x3

polyfit_results5 = np.exp(polyfit_quiffs4[1])*np.exp(polyfit_quiffs4[0]*x3)


x_queck = np.linspace(0.01,0.99,100)
y_queck = np.exp(polyfit_quiffs4[1])*np.exp(polyfit_quiffs4[0]*x_queck) 

plt.figure()
plt.plot(x3,y3,'-*k',x3,polyfit_results5,'-og',x_queck,y_queck,'-.b')
plt.xlabel('Normalised V2V Euclidean distance')
plt.ylabel('Likelihood of LOS')
plt.title('quiffs: A=%4f, B=%4f (note, y=A*exp(Bx)' % (polyfit_quiffs4[1],polyfit_quiffs4[0]))
plt.show()
plt.savefig('Expo-Fitted_LOS_V2V_Urban_Model.png')



# go crazy? Hdawg stylz.... code dome optimissation????
"""
seems to yield half-decent results...
In [30]: polyfit_quiffs4                                                                                 
Out[30]: array([-2.36945344, -0.08361897])








Look it was a valiant attempt none the fucking less.

good riddance.


from scipy.optimize import minimize

def orca_function(x):
    rome_los = np.array([0.93835616, 0.69970845, 0.44897959, 0.37037037, 0.26791277,
       0.21727019, 0.19512195, 0.17894737, 0.125     , 0.17665131])
    sf_los = np.array([0.9047619 , 0.62227913, 0.45698427, 0.29041916, 0.27157514,
       0.18581638, 0.18428438, 0.14766273, 0.15415306, 0.1116246 ])
    dist_array = np.array([0.05263158, 0.15789474, 0.26315789, 0.36842105, 0.47368421,
       0.57894737, 0.68421053, 0.78947368, 0.89473684,1])
    


    #COMPUTER GUESS:
    computer_guess = x[0]*np.exp(-1*x[1]*dist_array)

    #ROME ERROR: absolute_error = abs(rome_los-computer_guess)
    rome_error = abs(computer_guess-rome_los)

    #SF ERROR: 
    sf_error = abs(computer_guess-sf_los)

    #combined error?
    final_error_score = (rome_error+sf_error)**2


    return final_error_score


x0 = np.array([0.36667,0.9])
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

#res = minimize(orca_function, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, options={'xtol': 1e-7, 'disp':True})

res = minimize(orca_function, x0, method='BFGS', jac=rosen_der, options={'xtol': 1e-7, 'disp':True})



plt.figure()
plt.plot(normalised_distances, test,'-*g', normalised_distances,p_v2v_sf_los,'-sk',normalised_distances,p_v2v_rome_los,'-or')
plt.xlabel('V2V Euclidean distance')
plt.ylabel('Likelihood of LOS')
plt.title('why bother?')
"""

