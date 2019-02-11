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


CITY_NAME = 'Rome'
NUM_TAXIS = 300
#Roma geometries...
MIN_LAT = 41.856 
MAX_LAT = 41.928 
MIN_LON = 12.442 
MAX_LON = 12.5387 

#http://spatialreference.org/ref/epsg/32633/
#inProj = Proj(init='epsg:4326')
#outProj = Proj(init='epsg:32633')

##############################################################################################

slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LON,MAX_LON,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)


input_results_path = '/home/user/ABMTANET/maximal_connectedness/real_dataset_VANET_LCC_results/%s/%sish_taxis/' % (CITY_NAME, NUM_TAXIS)


output_filename = '%s_%i_taxis_LCC_data.pickle' % (CITY_NAME, NUM_TAXIS)


with open((input_results_path+output_filename), 'rb') as handle1:
    lcc_data_dict = pickle.load(handle1)
handle1.close()




plt.figure()
plt.plot(lcc_data_dict['timestamps'],lcc_data_dict['lcc_area_m2'],'-sr')
plt.xlabel('Time/[s]')
plt.ylabel('Area coverage/[m^2]')
plt.title(' %s %i Convex Hull Area' % (CITY_NAME, NUM_TAXIS))
plt.show()


H3, xedges3, yedges3 = np.histogram2d(lcc_data_dict['lcc_cofm_longs'], lcc_data_dict['lcc_cofm_lats'], bins=(Xedges,Yedges))
plt.figure()
plt.imshow(np.rot90(H3), cmap=plt.cm.BuPu_r)
plt.title('%s, num. Taxis:%i, LCC CofM Heatmap' % (CITY_NAME, NUM_TAXIS))
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
#plt.savefig((output_results_path+ ('%s_%i_taxis_vanet_heatmap_plot_TS_%i.png' % (CITY_NAME, NUM_TAXIS,timestamp))), dpi=300)
plt.show()


plt.figure()
plt.plot(lcc_data_dict['lcc_cofm_longs'], lcc_data_dict['lcc_cofm_lats'],'sr')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(' %s %i CofM Convex Hull' % (CITY_NAME, NUM_TAXIS))
plt.show()



plt.figure()
plt.plot(lcc_data_dict['timestamps'],lcc_data_dict['lcc_num_edges'],'--sr')
plt.plot(lcc_data_dict['timestamps'], lcc_data_dict['lcc_num_nodes'], '-or')
plt.xlabel('Time/[s]')
plt.ylabel('Number of things')
plt.title('%s %i LCC nodes and edges' % (CITY_NAME, NUM_TAXIS))
plt.legend(['Edges','Nodes'])
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



