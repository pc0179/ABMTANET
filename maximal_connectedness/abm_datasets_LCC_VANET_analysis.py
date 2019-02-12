"""
this is a complementary script to 'real_datasets_LCC_VANET_analysis.py'

in that it does the same job but uses ABM generated datasets to analyse
"""


import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pyproj import Proj, transform
#from shapely.geometry import Polygon

#plt.ion()
plt.ioff()

#On C207


"""
#SF ------------------------------------------------------------------------------------------- San Francisco!
CITY_NAME = 'SF'

# 2000+ taxis
sim_data_path = '/home/user/ABMTANET/maximal_connectedness/abm_sim_data/'

NUM_TAXIS = 300
abm_VANET_dataset_filename = 'sf_vanet_data_dict_300taxis_0.5hrs_v2vdata_6feb.pickle'
abm_positions_dataset_filename = 'sf_vanet_node_loc_dict_300taxis_0.5hrs_v2vdata_6feb.pickle'

#NUM_TAXIS = 2000
#abm_VANET_dataset_filename = 'sf_vanet_data_dict_2000taxis_0.5hrs_v2vdata_5feb.pickle'
#abm_positions_dataset_filename = 'sf_vanet_node_loc_dict_2000taxis_0.5hrs_v2vdata_5feb.pickle'


#output location
output_results_path = '/home/user/ABMTANET/maximal_connectedness/ABM_datasets_VANET_LCC_results/%s/%sish_taxis/' % (CITY_NAME, NUM_TAXIS)


# load the data in.
with open((sim_data_path+abm_VANET_dataset_filename), 'rb') as handle0:
    vanet_dict = pickle.load(handle0)
handle0.close()

with open((sim_data_path+abm_positions_dataset_filename), 'rb') as handle1:
    pos_dict = pickle.load(handle1)
handle1.close()


#SF geometries...
MIN_LAT = 37.733
MAX_LAT = 37.805 
MIN_LON = -122.479 
MAX_LON = -122.388 

inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:32610')
"""


##############################################################################################

#Rome ------------------------------------------------------------------------------------------- Rome!
CITY_NAME = 'Rome'

sim_data_path = '/home/user/ABMTANET/maximal_connectedness/abm_sim_data/'

#NUM_TAXIS = 300
#abm_VANET_dataset_filename = 'Roma_vanet_data_dict_300taxis_0.5hr_v2vdata_12feb.pickle'
#abm_positions_dataset_filename = 'Roma_vanet_node_loc_dict_300taxis_0.5hr_v2vdata_12feb.pickle' 

NUM_TAXIS = 2000
abm_VANET_dataset_filename = 'Roma_vanet_data_dict_2000taxis_0.5hr_v2vdata_12feb.pickle'
abm_positions_dataset_filename = 'Roma_vanet_node_loc_dict_2000taxis_0.5hr_v2vdata_12feb.pickle'


#output location
output_results_path = '/home/user/ABMTANET/maximal_connectedness/ABM_datasets_VANET_LCC_results/%s/%sish_taxis/' % (CITY_NAME, NUM_TAXIS)

# load the data in.
with open((sim_data_path+abm_VANET_dataset_filename), 'rb') as handle0:
    vanet_dict = pickle.load(handle0)
handle0.close()

with open((sim_data_path+abm_positions_dataset_filename), 'rb') as handle1:
    pos_dict = pickle.load(handle1)
handle1.close()


#Roma geometries...
MIN_LAT = 41.856 
MAX_LAT = 41.928 
MIN_LON = 12.442 
MAX_LON = 12.5387 

#http://spatialreference.org/ref/epsg/32633/
inProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:32633')

##############################################################################################



slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LON,MAX_LON,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)

convex_hull_area_m2_list = []
convex_hull_perimeter_m_list = []
num_LCC_nodes_list = []
num_LCC_edges_list = [] 
LCC_edge_density_list = [] 
LCC_node_area_coverage_list = [] 
LCC_CofM_longitude_list = [] 
LCC_CofM_latitude_list = []
LCC_taxi_id_set_dict = dict()

timestamp_list = sorted(list(vanet_dict.keys()))

starting_timestamp_index = 800
ending_timestamp_index = 1800
dt = 10
ts_index = list(range(starting_timestamp_index,ending_timestamp_index,dt))

for timestamp in ts_index:

    G = nx.Graph()
    #edges_list = [] #so edge-y!
    #edges_list = list(zip(vanet_dict[timestamp]['taxiAid'],vanet_dict[timestamp]['taxiBid']))

    G.add_edges_from(vanet_dict[timestamp]['edge_list'])

    G_lcc = max(nx.connected_component_subgraphs(G), key=len)  
    LCC_list = list(G_lcc.nodes())
    num_taxis_in_LCC = len(LCC_list)

    LCC_latitudes_array = np.zeros(num_taxis_in_LCC)
    LCC_longitudes_array = np.zeros(num_taxis_in_LCC)
    LCC_taxi_ids_array = np.zeros(num_taxis_in_LCC)

    lcc_taxi_counter = 0

    for taxi_id in LCC_list:

        LCC_latitudes_array[lcc_taxi_counter] = pos_dict[timestamp][taxi_id][1]
        LCC_longitudes_array[lcc_taxi_counter] = pos_dict[timestamp][taxi_id][0]
        lcc_taxi_counter +=1

    long_cofm_cc = np.mean(LCC_longitudes_array)
    lat_cofm_cc = np.mean(LCC_latitudes_array)

    lcc_min_lat =  LCC_latitudes_array.min() #min(LCC_latitudes_list)
    lcc_max_lat =  LCC_latitudes_array.max() # max(LCC_latitudes_list)
    lcc_min_long = LCC_longitudes_array.min() #min(LCC_longitudes_list)
    lcc_max_long = LCC_longitudes_array.max() #max(LCC_longitudes_list)


    utm_x, utm_y = transform(inProj, outProj, LCC_longitudes_array, LCC_latitudes_array)
    lcc_points_array = np.transpose(np.vstack((utm_x,utm_y)))
    lcc_hull = ConvexHull(lcc_points_array)
    hull_x_plotting_points = np.hstack((LCC_longitudes_array[lcc_hull.vertices], LCC_longitudes_array[lcc_hull.vertices[0]]))
    hull_y_plotting_points = np.hstack((LCC_longitudes_array[lcc_hull.vertices],LCC_longitudes_array[lcc_hull.vertices[0]]))
    

    #data collection...
    convex_hull_area_m2_list.append(lcc_hull.volume)
    convex_hull_perimeter_m_list.append(lcc_hull.area)

    num_LCC_nodes_list.append(num_taxis_in_LCC)
    num_LCC_edges_list.append(len(list(G_lcc.edges())))

    LCC_edge_density_list.append(num_LCC_edges_list[-1]/(num_LCC_nodes_list[-1]*(num_LCC_nodes_list[-1]-1)))
    LCC_node_area_coverage_list.append(convex_hull_area_m2_list[-1]/num_LCC_nodes_list[-1])

    LCC_CofM_longitude_list.append(long_cofm_cc)
    LCC_CofM_latitude_list.append(lat_cofm_cc)

    LCC_taxi_id_set_dict[timestamp] = set(LCC_list)

    
    #~~~~~~~~ Minor isues.... ~~~~~~##############################
    #okay... some differences here... previously, the real datasets would show locations of ALL taxis in the FLEET being simulated, whereas this is for only those that are communicating... hmmm.....
    
    H2, xedges2, yedges2 = np.histogram2d(vanet_dict[timestamp]['longitude'], vanet_dict[timestamp]['latitude'], bins=(Xedges,Yedges))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #PLotting....
    plt.figure()
    plt.imshow(np.rot90(H2), cmap=plt.cm.BuPu_r)
    plt.title('%s, num. Taxis:%i, time_step:%i, max_component: %i, total v2v exchanges: %i' % (CITY_NAME, NUM_TAXIS, timestamp, num_taxis_in_LCC, len(list(G.nodes()))))
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.title('Heatmap, TS: %i, num. taxis in lCC: %i' % (timestamp, num_taxis_in_LCC))
    plt.savefig((output_results_path+ ('%s_%i_taxis_vanet_heatmap_plot_TS_%i.png' % (CITY_NAME, NUM_TAXIS,timestamp))), dpi=300)
#    plt.show()
    

    plt.figure()
    plt.plot([lcc_min_long, lcc_min_long, lcc_max_long, lcc_max_long], [lcc_min_lat,lcc_max_lat,lcc_min_lat,lcc_max_lat], 'sk')
    plt.plot(LCC_longitudes_array, LCC_latitudes_array, 'ob')
    plt.plot(long_cofm_cc, lat_cofm_cc, 'dr')
    plt.plot(hull_x_plotting_points,hull_y_plotting_points,'*g--')
    plt.ylim([MIN_LAT, MAX_LAT])
    plt.xlim([MIN_LON, MAX_LON])
    plt.title('TS: %i, num. taxis in LCC: %i, LCC norm. area = %f' % (timestamp, num_taxis_in_LCC, (((convex_hull_area_m2_list[-1])/1000**2)/8**2)))
    plt.savefig((output_results_path+('%s_ABM_dataset_%i_taxis_vanet_lcc_plot_TS_%i.png' % (CITY_NAME, NUM_TAXIS,timestamp))), dpi=300)
    #Overall VANET heatmap...
    #H, xedges2, yedges2 = np.histogram2d(

    print('timestamp= %i, LCC_area= %f LCC_perimeter= %f, P/A= %f' % (timestamp, lcc_hull.volume, lcc_hull.area, (lcc_hull.area/lcc_hull.volume)))
    #quick check.. aka a 'queck'...
    print("graph_nodes: %i, graph_edges: %i" % (num_LCC_nodes_list[-1], num_LCC_edges_list[-1]))




####################### Output data gathering....

lcc_data_dict = {'NUM_TAXIS': NUM_TAXIS, 'timestamps':np.array(ts_index]), 'lcc_area_m2': np.array(convex_hull_area_m2_list), 'lcc_permiter_m': np.array(convex_hull_perimeter_m_list), 'lcc_cofm_lats': np.array(LCC_CofM_latitude_list), 'lcc_cofm_longs': LCC_CofM_longitude_list, 'lcc_num_edges': np.array(num_LCC_edges_list), 'lcc_num_nodes': np.array(num_LCC_nodes_list), 'lcc_node_area_cov': np.array(LCC_node_area_coverage_list), 'lcc_edge_density': np.array(LCC_edge_density_list)}


output_filename = '%s_%i_taxis_LCC_ABM_data.pickle' % (CITY_NAME, NUM_TAXIS)
with open((output_results_path +output_filename), 'wb') as handle9:
    pickle.dump(lcc_data_dict, handle9, protocol=pickle.HIGHEST_PROTOCOL)
handle9.close()


output_filename2 = '%s_%i_taxis_LCC_taxi_ids_ABM_dict.pickle' % (CITY_NAME, NUM_TAXIS)
with open((output_results_path +output_filename2), 'wb') as handle10:
    pickle.dump(LCC_taxi_id_set_dict, handle10, protocol=pickle.HIGHEST_PROTOCOL)
handle10.close()


# More plots.....







