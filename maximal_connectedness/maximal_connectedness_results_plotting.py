"""
ever more desperate.
ever more pointless.
yet here we are coding...

attempt to understand maximal connected components of VANET's 

data currently from ABM, SF, 2000 taxis, 30 minutes of simulation.

"""

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#plt.ion()
plt.ioff()



#on C207
#SF ABM DATA, 2k taxis, 30 mins
sim_data_path = '/home/user/ABMTANET/maximal_connectedness/abm_sim_data/'
#sim_results_filename = 'sf_vanet_data_dict_1000taxis_0.5hrs_v2vdata_4feb.pickle'


vanet_loc_data_filename = 'sf_vanet_node_loc_dict_2000taxis_0.5hrs_v2vdata_5feb.pickle'
vanet_graph_data_filename = 'sf_vanet_data_dict_2000taxis_0.5hrs_v2vdata_5feb.pickle'



with open((sim_data_path+vanet_loc_data_filename), 'rb') as handle0:
    vanet_pos_list_of_dicts = pickle.load(handle0)
handle0.close()

with open((sim_data_path+vanet_graph_data_filename), 'rb') as handle1:
    vanet_graph_data_dict = pickle.load(handle1)
handle1.close()


CITY_NAME = 'SF'
NUM_TAXIS = 1000

MIN_LAT = 37.733 #general_model_params_dict['min_lat']
MAX_LAT = 37.805 #general_model_params_dict['max_lat']
MIN_LON = -122.479 #general_model_params_dict['min_lon']
MAX_LON = -122.388 #general_model_params_dict['max_lon']

slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LON,MAX_LON,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)


ts_list = list(vanet_graph_data_dict.keys()) 
output_results_path = '/home/user/ABMTANET/maximal_connectedness/VANET_LCC_results/'

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

# ISSUES HERE, note that largest_cc is a list of the TAXI ID's, not an index... hence the following couple of lines don't make sense...
    #lcc_long_x = vanet_dict[i]['longitude'][largest_cc]
    #lcc_lat_y = vanet_dict[i]['latitude'][largest_cc]
############################
# will need to get actual locations of these damn vehicles.... 


    long_cofm_cc = np.mean(taxi_lon_array)
    lat_cofm_cc = np.mean(taxi_lat_array)

    lcc_min_lat = min(taxi_lat_array)
    lcc_max_lat = max(taxi_lat_array)
    lcc_min_long = min(taxi_lon_array)
    lcc_max_long = max(taxi_lon_array)

    H2, xedges2, yedges2 = np.histogram2d(vanet_graph_data_dict[i]['longitude'], vanet_graph_data_dict[i]['latitude'], bins=(Xedges,Yedges))


    plt.figure()
    plt.imshow(np.rot90(H2), cmap=plt.cm.BuPu_r)
    plt.title('%s, num. Taxis:%i, time_step:%i, max_component: %i, total v2v exchanges: %i' % (CITY_NAME, NUM_TAXIS, i, num_taxis_in_largest_cc, len(vanet_graph_data_dict[i]['edge_list'])))
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.title('Heatmap, TS: %i, num. taxis in lCC: %i' % (i, num_taxis_in_largest_cc))
    plt.savefig((output_results_path+ ('vanet_heatmap_plot_TS_%i.png' % (i))), dpi=300)
#    plt.show()

    plt.figure()
    plt.plot([lcc_min_long, lcc_min_long, lcc_max_long, lcc_max_long], [lcc_min_lat,lcc_max_lat,lcc_min_lat,lcc_max_lat], 'sk')
    plt.plot(taxi_lon_array, taxi_lat_array, 'ob')
    plt.plot(long_cofm_cc, lat_cofm_cc, 'dr')
    plt.ylim([MIN_LAT, MAX_LAT])
    plt.xlim([MIN_LON, MAX_LON])
    plt.title('TS: %i, num. taxis in lCC: %i' % (i, num_taxis_in_largest_cc))
    plt.savefig((output_results_path+('vanet_lcc_plot_TS_%i.png' % (i))), dpi=300)
#    plt.show()



















######################################################################################################

"""
and now for the REAL taxi trace datasets!
all still on c207

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#plt.ion()
plt.ioff()


"""

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#plt.ion()
plt.ioff()


real_taxi_trace_dataset_path = '/home/user/ClusterSetUp-All-to-One/SF-Data/'
real_taxi_trace_dataset_filename = 'No_Pandas_CORRECTED_SF_9_day_VANET.pickle'
 #'No_Pandas_sf_taxis_positions_combined_9_days.pickle'
real_taxi_trace_output_results_path = '/home/user/ABMTANET/maximal_connectedness/real_trace_VANET_LCC_results/'


with open((real_taxi_trace_dataset_path+real_taxi_trace_dataset_filename), 'rb') as handle2:
    real_trace_vanet_dict = pickle.load(handle2)
handle2.close()


CITY_NAME = 'SF'
NUM_TAXIS = 1000

MIN_LAT = 37.733 #general_model_params_dict['min_lat']
MAX_LAT = 37.805 #general_model_params_dict['max_lat']
MIN_LON = -122.479 #general_model_params_dict['min_lon']
MAX_LON = -122.388 #general_model_params_dict['max_lon']

slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LON,MAX_LON,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)


ts_list = list(real_trace_vanet_dict.keys()) 

for i in range(ts_list[100], ts_list[1100],10):

    edge_list = []
    edge_list = list(zip(real_trace_vanet_dict[i]['taxiAid'],real_trace_vanet_dict[i]['taxiBid']))

    G = nx.Graph()
    G.add_edges_from(edge_list)

    largest_cc = list(max(nx.connected_components(G), key=len))
    num_taxis_in_largest_cc = len(largest_cc)

    taxi_lat_array = np.zeros(num_taxis_in_largest_cc)
    taxi_lon_array = np.zeros(num_taxis_in_largest_cc)
    for j in range(0,num_taxis_in_largest_cc):

        if largest_cc[j] in real_trace_vanet_dict[i]['taxiAid']:
            index_id = real_trace_vanet_dict[i]['taxiAid'].index(largest_cc[j])
            taxi_lon_array[j] = real_trace_vanet_dict[i]['Alonglat'][index_id][0]
            taxi_lat_array[j] = real_trace_vanet_dict[i]['Alonglat'][index_id][1]
        
        else:
            index_id = real_trace_vanet_dict[i]['taxiBid'].index(largest_cc[j])
            taxi_lon_array[j] = real_trace_vanet_dict[i]['Blonglat'][index_id][0]
            taxi_lat_array[j] = real_trace_vanet_dict[i]['Blonglat'][index_id][1]      


# ISSUES HERE, note that largest_cc is a list of the TAXI ID's, not an index... hence the following couple of lines don't make sense...
    #lcc_long_x = vanet_dict[i]['longitude'][largest_cc]
    #lcc_lat_y = vanet_dict[i]['latitude'][largest_cc]
############################
# will need to get actual locations of these damn vehicles.... 


    long_cofm_cc = np.mean(taxi_lon_array)
    lat_cofm_cc = np.mean(taxi_lat_array)

    lcc_min_lat = min(taxi_lat_array)
    lcc_max_lat = max(taxi_lat_array)
    lcc_min_long = min(taxi_lon_array)
    lcc_max_long = max(taxi_lon_array)

    H2, xedges2, yedges2 = np.histogram2d(taxi_lon_array, taxi_lat_array, bins=(Xedges,Yedges))


    plt.figure()
    plt.imshow(np.rot90(H2), cmap=plt.cm.BuPu_r)
    plt.title('%s, num. Taxis:%i, time_step:%i, max_component: %i, total v2v exchanges: %i' % (CITY_NAME, NUM_TAXIS, i, num_taxis_in_largest_cc, len(edge_list)))
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.title('Heatmap, TS: %i, num. taxis in lCC: %i' % (i, num_taxis_in_largest_cc))
    plt.savefig((real_taxi_trace_output_results_path + ('%s_real_trace_data_vanet_heatmap_plot_TS_%i.png' % (CITY_NAME,i))), dpi=300)
#    plt.show()

    plt.figure()
    plt.plot([lcc_min_long, lcc_min_long, lcc_max_long, lcc_max_long], [lcc_min_lat,lcc_max_lat,lcc_min_lat,lcc_max_lat], 'sk')
    plt.plot(taxi_lon_array, taxi_lat_array, 'ob')
    plt.plot(long_cofm_cc, lat_cofm_cc, 'dr')
    plt.ylim([MIN_LAT, MAX_LAT])
    plt.xlim([MIN_LON, MAX_LON])
    plt.title('TS: %i, num. taxis in lCC: %i' % (i, num_taxis_in_largest_cc))
    plt.savefig((real_taxi_trace_output_results_path+('%s_real_trace_data_vanet_lcc_plot_TS_%i.png' % (CITY_NAME, i))), dpi=300)
#    plt.show()

























