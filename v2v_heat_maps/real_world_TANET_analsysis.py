"""
24Jan2019

script to explore the 'real world' VANET datasets...
currently focusing on:
- heat maps of where V2V exchanges occur...
- number of 2 hop neighbours...
- total number of exchanges...

given they are 8*8km blocs, maybe use 50*50 metres to sample the data?

p.s. bear in mind, that the 'real world' data was sampled at 10s... hmmmmmm

"""


import pickle
import numpy as np


# currently on C207
"""
# ROMA
real_tanet_data_filepath = '/home/user/ClusterSetUp-All-to-One/Rome-Data/'
rome_real_tanet_data_files = ['No_Pandas_VANET_rome_august_edit_combined_3_days.pickle', 'No_Pandas_VANET_rome_august_edit_combined_7_days.pickle', 'No_Pandas_VANET_august_edit_combined_14_days_rome.pickle', 'split_dict_15_till_20_hrs_No_Pandas_VANET_Rome_21_days.pickle','Corrected_no_pandas_vanet_rome_29_days.pickle']
rome_numfolded_days = [3,7,14,21,29]
#output_filepath = '/home/user/ABMTANET/v2v_heat_maps/rome/'
output_filepath = '/home/user/ABMTANET/two_hop_count/rome/'
"""
# San Francisco
real_tanet_data_filepath = '/home/user/ClusterSetUp-All-to-One/SF-Data/'
sf_real_tanet_data_files = ['No_Pandas_CORRECTED_SF_1_day_VANET.pickle', 'No_Pandas_CORRECTED_VANET_SF_2_days.pickle', 'No_Pandas_CORRECTED_SF_4_days_VANET.pickle', 'No_Pandas_CORRECTED_VANET_SF_6_days.pickle','No_Pandas_CORRECTED_SF_9_day_VANET.pickle']
sf_numfolded_days = [1,2,4,6,9]
#output_filepath = '/home/user/ABMTANET/v2v_heat_maps/rome/'
output_filepath = '/home/user/ABMTANET/two_hop_count/SF/'




"""
#V2V EXCHANGE HEAT MAP ANALYSIS

# purely to get the longitudes/latitudes needed for 2d histogram...
CITY_NAME = 'Roma'
SIM_RUN_DATE = '2Ktaxis_30mins_expolos_23jan'
sim_data_results_filepath = '/home/user/ABMTANET/simulation_results/'
general_model_params_filename = '%s_general_model_params_%s.pickle' % (CITY_NAME,SIM_RUN_DATE)

with open((sim_data_results_filepath+general_model_params_filename),'rb') as handle3:
    general_model_params_dict = pickle.load(handle3)

MIN_LAT = general_model_params_dict['min_lat']
MAX_LAT = general_model_params_dict['max_lat']
MIN_LON = general_model_params_dict['min_lon']
MAX_LON = general_model_params_dict['max_lon']

handle3.close()


#comparing heatmaps between 'real' rome traces and taxi-agent traces....



internal_file_counter = 0
slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LON,MAX_LON,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)

for dataset in rome_real_tanet_data_files:

    with open(real_tanet_data_filepath+dataset, 'rb') as handle0:
        real_taxi_data_vanet_dict = pickle.load(handle0)


    sum_real_taxi_vanet_H2 = 0
# once again, finding the mid-points of the 'real' V2V exchanges....
#minor memory issues... total 'real' data is big... 600mb compressed?
    for key, values in real_taxi_data_vanet_dict.items():

        Alon, Alat = np.array(list(zip(*values['Alonglat'])))
        Blon, Blat = np.array(list(zip(*values['Blonglat'])))

        v2v_mid_point_longitude_array = (Alon+Blon)/2
        v2v_mid_point_latitude_array = (Alat+Blat)/2 
        H2, xedges2, yedges2 = np.histogram2d(v2v_mid_point_longitude_array, v2v_mid_point_latitude_array, bins=(Xedges,Yedges))

        sum_real_taxi_vanet_H2 += H2

    #total_real_v2v_exchanges = 0
    #for i in range(sum_real_taxi_vanet_H2.shape[0]):
        #total_real_v2v_exchanges += np.sum(sum_real_taxi_vanet_H2[i])


    output_filename = 'rome_%i_days_v2v_heatmap_data.pickle' % rome_numfolded_days[internal_file_counter]
    internal_file_counter +=1


    #save processed V2V network data...

    print(dataset)
    print(sum_real_taxi_vanet_H2)

    with open((output_filepath + output_filename), 'wb') as handle1:
        pickle.dump( sum_real_taxi_vanet_H2 , handle1, protocol=pickle.HIGHEST_PROTOCOL)

    #free up RAM?
    handle0.close()
    handle1.close()


"""




import networkx as nx

internal_file_counter = 0
n=2
two_hop_count_dict = dict()

for dataset in sf_real_tanet_data_files:

    with open(real_tanet_data_filepath+dataset, 'rb') as handle0:
        real_taxi_data_vanet_dict = pickle.load(handle0)


# once again, finding the mid-points of the 'real' V2V exchanges....
#minor memory issues... total 'real' data is big... 600mb compressed?
    for key, values in real_taxi_data_vanet_dict.items():

        edge_pair_list = list(zip(values['taxiAid'], values['taxiBid']))
        G = nx.Graph()
        G.add_edges_from(edge_pair_list)

        taxi_nodes_list = list(G.nodes)
        two_hop_count_list = []

        for taxi_node in taxi_nodes_list:
            path_lengths = nx.single_source_dijkstra_path_length(G, taxi_node)
            two_hop_count_list.append(len([taxi_node for taxi_node, length in path_lengths.items() if length == n])) #ie the number of taxis accessible within two hops...? could be interesting

        two_hop_count_dict[key] = np.array(two_hop_count_list)

    output_filename = 'rome_%i_days_two_hop_count.pickle' % sf_numfolded_days[internal_file_counter]
    internal_file_counter +=1

    #save processed V2V network data...

    print(dataset)
    print(two_hop_count_list)

    with open((output_filepath + output_filename), 'wb') as handle1:
        pickle.dump( two_hop_count_dict , handle1, protocol=pickle.HIGHEST_PROTOCOL)

    #free up RAM?
    handle0.close()
    handle1.close()








"""
fig, ax1 = plt.subplots(1,1)
c = ax1.pcolor(sum_real_taxi_vanet_H2)
ax1.set_title('real... V2V Exchange Heat-map, %s, total: %i, NUM of real TAXIS: 2250' % (CITY_NAME, int(total_real_v2v_exchanges)))
fig.tight_layout()
plt.show()







# for inspiration later.... 
def LoSHopCount(df,n):

    edge_pair_list = list(zip(df.taxiAid[df.num_buildings<1],df.taxiBid[df.num_buildings<1]))

    G = nx.Graph()
    G.add_edges_from(edge_pair_list)

    taxi_nodes_list = list(G.nodes)
    two_hop_count_list = []
    #col_name = ('%ihop' % (n))
    for taxi_node in taxi_nodes_list:

        path_lengths = nx.single_source_dijkstra_path_length(G,taxi_node)
        two_hop_count_list.append(len([taxi_node for taxi_node, length in path_lengths.items() if length == n]))

    return two_hop_count

"""

