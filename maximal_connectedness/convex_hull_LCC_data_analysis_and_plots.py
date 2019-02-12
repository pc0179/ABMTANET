"""
this script attempts to plot data outputed from 'real_datasets_LCC_VANET_analysis_plots.py'

essentially computed results data regarding convex hulls of VANETS (in real cities)


"""

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#from scipy.spatial import ConvexHull
#from pyproj import Proj, transform
#from shapely.geometry import Polygon

plt.ion()
#plt.ioff()


"""
#Roma
CITY_NAME = 'Rome'
NUM_TAXIS = 2000 #300
DATA_TYPE = 'ABM'
#Roma geometries...
MIN_LAT = 41.856 
MAX_LAT = 41.928 
MIN_LON = 12.442 
MAX_LON = 12.5387 
"""
##############################################################################################

#San Francisco ABM dataset analysis
CITY_NAME = 'SF'
NUM_TAXIS = 300 #2000
DATA_TYPE = 'ABM'


MIN_LAT = 37.733
MAX_LAT = 37.805 
MIN_LON = -122.479 
MAX_LON = -122.388 




slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LON,MAX_LON,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)


input_results_path = '/home/user/ABMTANET/maximal_connectedness/%s_datasets_VANET_LCC_results/%s/%sish_taxis/' % (DATA_TYPE, CITY_NAME, NUM_TAXIS)


output_filename = '%s_%i_taxis_LCC_%s_data.pickle' % (CITY_NAME, NUM_TAXIS, DATA_TYPE)


with open((input_results_path+output_filename), 'rb') as handle1:
    lcc_data_dict = pickle.load(handle1)
handle1.close()

####### minor issue regards timestamps...
starting_timestamp_index = 800
ending_timestamp_index = 1800
dt = 10
ts_index = list(range(starting_timestamp_index,ending_timestamp_index,dt))
lcc_data_dict['timestamps'] = ts_index

plt.figure()
plt.plot(lcc_data_dict['timestamps'],lcc_data_dict['lcc_area_m2']/(1000**2),'-sr')
plt.xlabel('Time/[s]')
plt.ylabel('Area coverage/[km^2]')
plt.title('%s %s %i Convex Hull Area' % (DATA_TYPE, CITY_NAME, NUM_TAXIS))
plt.savefig((input_results_path+'%s_%s_%i_taxis_LCC_area_coverage.png' %(DATA_TYPE, CITY_NAME, NUM_TAXIS)),dpi=300)
plt.show()


H3, xedges3, yedges3 = np.histogram2d(lcc_data_dict['lcc_cofm_longs'], lcc_data_dict['lcc_cofm_lats'], bins=(Xedges,Yedges))
plt.figure()
plt.imshow(np.rot90(H3), cmap=plt.cm.BuPu_r)
plt.title('%s %s, num. Taxis:%i, LCC CofM Heatmap' % (DATA_TYPE, CITY_NAME, NUM_TAXIS))
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.savefig((input_results_path+'%s_%s_%i_taxis_LCC_CofM_Heatmap.png' %(DATA_TYPE, CITY_NAME, NUM_TAXIS)),dpi=400)
plt.show()

"""
plt.figure()
plt.plot(lcc_data_dict['lcc_cofm_longs'], lcc_data_dict['lcc_cofm_lats'],'sr')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('%s %s %i CofM Convex Hull' % (DATA_TYPE, CITY_NAME, NUM_TAXIS))
plt.savefig((input_results_path+'%s_%s_%i_taxis_LCC_CofM_raw_plot.png' %(DATA_TYPE, CITY_NAME, NUM_TAXIS)),dpi=300)
plt.show()
"""


plt.figure()
plt.plot(lcc_data_dict['timestamps'],lcc_data_dict['lcc_num_edges'],'--sr')
plt.plot(lcc_data_dict['timestamps'], lcc_data_dict['lcc_num_nodes'], '-or')
plt.xlabel('Time/[s]')
plt.ylabel('Number of ...')
plt.title('%s %s %i LCC nodes and edges' % (DATA_TYPE, CITY_NAME, NUM_TAXIS))
plt.legend(['Edges','Nodes'])
plt.savefig((input_results_path+'%s_%s_%i_taxis_LCC_nodes_edges_quantity.png' %(DATA_TYPE, CITY_NAME, NUM_TAXIS)),dpi=300)
plt.show()




plt.figure()
plt.plot(lcc_data_dict['timestamps'], lcc_data_dict['lcc_node_area_cov'], '-ob')
plt.xlabel('Time/[s]')
plt.ylabel('Mean node-area-coverage/[m^2]')
plt.title('%s %s %i LCC node area coverage' %(DATA_TYPE, CITY_NAME, NUM_TAXIS))
plt.savefig((input_results_path+'%s_%s_%i_taxis_LCC_node_area_coverage.png' %(DATA_TYPE, CITY_NAME, NUM_TAXIS)),dpi=300)
plt.show()


plt.figure()
plt.plot(lcc_data_dict['timestamps'], lcc_data_dict['lcc_num_edges']/(lcc_data_dict['lcc_num_nodes']*(lcc_data_dict['lcc_num_nodes']-1)), '-ob')
plt.xlabel('Time/[s]')
plt.ylabel('LCC Edge Density (connectedness?)')
plt.title('%s %s %i LCC edge density/ graph connectedness' %(DATA_TYPE, CITY_NAME, NUM_TAXIS))
plt.savefig((input_results_path+'%s_%s_%i_taxis_LCC_graph_connectedness.png' %(DATA_TYPE, CITY_NAME, NUM_TAXIS)),dpi=300)
plt.show()


######## set mixing investigation....

output_filename2 = '%s_%i_taxis_LCC_taxi_ids_%s_dict.pickle' % (CITY_NAME, NUM_TAXIS, DATA_TYPE)
with open((input_results_path+output_filename2), 'rb') as handle2:
    lcc_taxi_id_dict = pickle.load(handle2)
handle2.close()






"""
set_1 = lcc_taxi_id_dict[68520]
set_2 = lcc_taxi_id_dict[68530]

interset_12 = set_1.intersection(set_2)
interset_21 = set_2.intersection(set_1)

setdiff_interset122 = interset_12.difference(set_2)

setdiff_2interset12 = set_2.difference(interset_12)

"""

mixing_quiff = []
set_len_list = []
timestamps_list = sorted(list(lcc_taxi_id_dict.keys()))
dt = 10 # the timestep between timestamps... is ten seconds for real datasets...
#for timestamps, values in lcc_taxi_id_dict.items():

for t in timestamps_list[1:]:

    intersect_taxis_BA = lcc_taxi_id_dict[t].intersection(lcc_taxi_id_dict[t-dt])
    len_setA = len(lcc_taxi_id_dict[t-dt])
    mixing_quiff.append(len(intersect_taxis_BA)/len_setA)
    set_len_list.append(len(lcc_taxi_id_dict[t]))



plt.figure()
plt.plot(timestamps_list[1:],mixing_quiff,'-sr')
plt.plot(timestamps_list[1:], np.array(set_len_list)/NUM_TAXIS,'--ob')
plt.xlabel('Time/[s]')
#plt.ylabel('Mixing Coeff. len(intersectionAB)/len(A)')
plt.title('%s %s, %i taxis LCC mixing coef.' % (DATA_TYPE, CITY_NAME, NUM_TAXIS))
plt.legend(['Similarity Coeff. with previous LCC', 'Num. in LCC/Fleet Size'])
plt.savefig((input_results_path+'%s_%s_%i_taxis_LCC_mixing_coef.png' %(DATA_TYPE, CITY_NAME, NUM_TAXIS)),dpi=400)
plt.show()







"""
spare code....


lcc_data_dict = {'NUM_TAXIS': NUM_TAXIS, 'timestamps':np.array(timestamp_list[starting_timestamp_index :ending_timestamp_index]), 'lcc_area_m2': np.array(convex_hull_area_m2_list), 'lcc_permiter_m': np.array(convex_hull_perimeter_m_list), 'lcc_cofm_lats': np.array(LCC_CofM_latitude_list), 'lcc_cofm_longs': LCC_CofM_longitude_list, 'lcc_num_edges': np.array(num_LCC_edges_list), 'lcc_num_nodes': np.array(num_LCC_nodes_list), 'lcc_node_area_cov': np.array(LCC_node_area_coverage_list), 'lcc_edge_density': np.array(LCC_edge_density_list)}



#####
    #PLotting....
    plt.figure()
    plt.imshow(np.rot90(H2), cmap=plt.cm.BuPu_r)
    plt.title('%s, num. Taxis:%i, time_step:%i, max_component: %i, total v2v exchanges: %i' % (CITY_NAME, NUM_TAXIS, timestamp, num_taxis_in_LCC, len(edges_list)))
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.title('Heatmap, TS: %i, num. taxis in lCC: %i' % (timestamp, num_taxis_in_LCC))
    plt.savefig((output_results_path+ ('%s_%i_taxis_vanet_heatmap_plot_TS_%i.png' % (CITY_NAME, NUM_TAXIS,timestamp))), dpi=300)
#    plt.show()


"""



