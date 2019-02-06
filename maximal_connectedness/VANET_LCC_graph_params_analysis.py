"""
sort of moving on slightly from maximal_connectedness_results_plotting.py

now focusing on more general graph theory parameters of interest

aiming to establish a stability of VANETs criterion.... in your phucking dreams...

from the simulations...

collect following raw data:
- number of taxis currently in fleet at timestep
- number of taxis within LCC
- number of edges in LCC
- CofM of LCC
- area of convex hull

compute following:
- fleet fraction in LCC
- LCC node density (num of nodes/convex hull area)
- graph density: edges/n(n-1)


"""

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import pyproj


from shapely.geometry import Polygon


#plt.ion()
plt.ioff()

#on C207

NUM_TAXIS = 2000

"""
#Rome ABM Data
CITY_NAME = 'Roma'
vanet_loc_data_filename = 'Roma_vanet_node_loc_dict_2000taxis_0.5hr_v2vdata_5feb.pickle'
vanet_graph_data_filename = 'Roma_vanet_data_dict_2000taxis_0.5hr_v2vdata_5feb.pickle'
MAX_LAT = 41.928
MIN_LAT = 41.856
MIN_LON = 12.442
MAX_LON = 12.5387
"""


#SF ABM DATA, 2k taxis, 30 mins
#sim_data_path = '/home/user/ABMTANET/maximal_connectedness/abm_sim_data/'
#sim_results_filename = 'sf_vanet_data_dict_1000taxis_0.5hrs_v2vdata_4feb.pickle'

CITY_NAME = 'SF'
sim_data_path = '/home/user/ABMTANET/maximal_connectedness/abm_sim_data/'
#sim_data_path = '/media/user/B53D-1DA8/6thfeb_sim_results/'
vanet_loc_data_filename = 'sf_vanet_node_loc_dict_2000taxis_0.5hrs_v2vdata_5feb.pickle'
vanet_graph_data_filename = 'sf_vanet_data_dict_2000taxis_0.5hrs_v2vdata_5feb.pickle'

#vanet_loc_data_filename  = 'sf_vanet_node_loc_dict_300taxis_0.5hrs_v2vdata_6feb.pickle'
#vanet_graph_data_filename = 'sf_vanet_data_dict_300taxis_0.5hrs_v2vdata_6feb.pickle'

MIN_LAT = 37.733 #general_model_params_dict['min_lat']
MAX_LAT = 37.805 #general_model_params_dict['max_lat']
MIN_LON = -122.479 #general_model_params_dict['min_lon']
MAX_LON = -122.388 #general_model_params_dict['max_lon']

wgs84 = pyproj.Proj("+init=EPSG:4326") # LatLon classic.
UTM10N = pyproj.Proj("+init=EPSG:32610") #urgh?



#debugging this phucking convex hull area.....
# cha*200 (200 seems to be the necessary correction factor....)
# so very lost... 
test_hull_array = np.array([[MIN_LON,MIN_LAT],[MIN_LON,MAX_LAT],[MAX_LON,MIN_LAT],[MAX_LON, MAX_LAT]])

x_test_hull_array = np.array([(MIN_LON+MAX_LON)/2,(MIN_LON+MAX_LON)/2,MAX_LON,MAX_LON])
y_test_hull_array = np.array([MIN_LAT,MAX_LAT,MIN_LAT,MAX_LAT])


utm_points_array = UTM10N(x_test_hull_array,y_test_hull_array)
lcc_points_array = np.transpose(np.vstack((utm_points_array[0],utm_points_array[1])))
lcc_hull = ConvexHull(lcc_points_array)
hull_x_plotting_points = x_test_hull_array[lcc_hull.vertices]
hull_y_plotting_points = y_test_hull_array[lcc_hull.vertices]

convex_hull_area_m2_test = lcc_hull.area

further_test_point = UTM10N(MAX_LON,MAX_LAT)

plt.figure(1111)
plt.plot(hull_x_plotting_points,hull_y_plotting_points,'*g--')
plt.ylim([MIN_LAT, MAX_LAT])
plt.xlim([MIN_LON, MAX_LON])
plt.show(1111)

test_points = lcc_points_array.tolist() #list(zip((lcc_points_array[0].tolist(), lcc_points_array[1].tolist())))
pshape = Polygon(test_points)

test_points2 = ((549900,4176350),(553900,4176350),(553900,418350),(549900,418350))
pshape = Polygon(test_points2)


#fuck my life.





















# Loadin' data...

slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LON,MAX_LON,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)


with open((sim_data_path+vanet_loc_data_filename), 'rb') as handle0:
    vanet_pos_list_of_dicts = pickle.load(handle0)
handle0.close()

with open((sim_data_path+vanet_graph_data_filename), 'rb') as handle1:
    vanet_graph_data_dict = pickle.load(handle1)
handle1.close()


output_results_path = '/home/user/ABMTANET/maximal_connectedness/VANET_LCC_results/%s/' % CITY_NAME


ts_list = list(vanet_graph_data_dict.keys()) 
convex_hull_area_m2_list = []
cchasf = 2000

for i in range(ts_list[100], ts_list[1100],10):

    G = nx.Graph()
    G.add_edges_from(vanet_graph_data_dict[i]['edge_list'])

    largest_cc = list(max(nx.connected_components(G), key=len))
    num_taxis_in_largest_cc = len(largest_cc)

    taxi_lat_array = np.zeros(num_taxis_in_largest_cc)
    taxi_lon_array = np.zeros(num_taxis_in_largest_cc)
    for j in range(0,num_taxis_in_largest_cc):
        taxi_lon_array[j] = vanet_pos_list_of_dicts[i][largest_cc[j]][0]
        taxi_lat_array[j] = vanet_pos_list_of_dicts[i][largest_cc[j]][1]

    long_cofm_cc = np.mean(taxi_lon_array)
    lat_cofm_cc = np.mean(taxi_lat_array)

    lcc_min_lat = min(taxi_lat_array)
    lcc_max_lat = max(taxi_lat_array)
    lcc_min_long = min(taxi_lon_array)
    lcc_max_long = max(taxi_lon_array)

    H2, xedges2, yedges2 = np.histogram2d(vanet_graph_data_dict[i]['longitude'], vanet_graph_data_dict[i]['latitude'], bins=(Xedges,Yedges))


    #convex hull...
    #will need to convert to metres in order to compute convex hull area.... justsayin'...

#    lcc_points = np.transpose(np.vstack((taxi_lon_array,taxi_lat_array)))
#    lcc_hull = ConvexHull(lcc_points)
#    hull_x_plotting_points = lcc_points[lcc_hull.vertices,0]
#    hull_y_plotting_points = lcc_points[lcc_hull.vertices,1]


    utm_points_array = UTM10N(taxi_lon_array, taxi_lat_array)
    lcc_points_array = np.transpose(np.vstack((utm_points_array[0],utm_points_array[1])))
    lcc_hull = ConvexHull(lcc_points_array)
    hull_x_plotting_points = np.hstack((taxi_lon_array[lcc_hull.vertices], taxi_lon_array[lcc_hull.vertices[0]]))
    hull_y_plotting_points = np.hstack((taxi_lat_array[lcc_hull.vertices],taxi_lat_array[lcc_hull.vertices[0]]))
    convex_hull_area_m2_list.append(lcc_hull.area)

    plt.figure()
    plt.imshow(np.rot90(H2), cmap=plt.cm.BuPu_r)
    plt.title('%s, num. Taxis:%i, time_step:%i, max_component: %i, total v2v exchanges: %i' % (CITY_NAME, NUM_TAXIS, i, num_taxis_in_largest_cc, len(vanet_graph_data_dict[i]['edge_list'])))
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.title('Heatmap, TS: %i, num. taxis in lCC: %i' % (i, num_taxis_in_largest_cc))
    plt.savefig((output_results_path+ ('%s_%i_taxis_vanet_heatmap_plot_TS_%i.png' % (CITY_NAME, NUM_TAXIS,i))), dpi=300)
#    plt.show()

    plt.figure()
    plt.plot([lcc_min_long, lcc_min_long, lcc_max_long, lcc_max_long], [lcc_min_lat,lcc_max_lat,lcc_min_lat,lcc_max_lat], 'sk')
    plt.plot(taxi_lon_array, taxi_lat_array, 'ob')
    plt.plot(long_cofm_cc, lat_cofm_cc, 'dr')
    plt.plot(hull_x_plotting_points,hull_y_plotting_points,'*g--')
    plt.ylim([MIN_LAT, MAX_LAT])
    plt.xlim([MIN_LON, MAX_LON])
    plt.title('TS: %i, num. taxis in LCC: %i, LCC norm. area = %f' % (i, num_taxis_in_largest_cc, (((convex_hull_area_m2_list[-1]*cchasf)/1000**2)/8**2)))
    plt.savefig((output_results_path+('%s_%i_taxis_vanet_lcc_plot_TS_%i.png' % (CITY_NAME, NUM_TAXIS,i))), dpi=300)
#    plt.show()

    print(i)


convex_hull_area_norm_array = (((np.array(convex_hull_area_m2_list))/1000**2)/8**2)*200



















