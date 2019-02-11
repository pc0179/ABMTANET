"""
Mon 1th Feb 2k19
shitshitshit
now conducing analysis on real taxi (folded) datasets... Rome and SF for comparison... to abm cases.

investigating properties of the Largest Connected Component of the VANET graph throughout simulation time period...

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
sim_data_path = '/home/user/ClusterSetUp-All-to-One/SF-Data/'

#NUM_TAXIS = 300
#real_VANET_dataset_filename = 'No_Pandas_CORRECTED_SF_1_day_VANET.pickle'
#real_positions_dataset_filename = 'No_Pandas_CORRECTED_sf_taxis_positions_combined_1_days.pickle' 

NUM_TAXIS = 2000
real_VANET_dataset_filename = 'No_Pandas_CORRECTED_SF_9_day_VANET.pickle'
real_positions_dataset_filename = 'No_Pandas_sf_taxis_positions_combined_9_days.pickle'


#output location
output_results_path = '/home/user/ABMTANET/maximal_connectedness/real_dataset_VANET_LCC_results/%s/%sish_taxis/' % (CITY_NAME, NUM_TAXIS)


# load the data in.
with open((sim_data_path+real_VANET_dataset_filename), 'rb') as handle0:
    vanet_dict = pickle.load(handle0)
handle0.close()

with open((sim_data_path+real_positions_dataset_filename), 'rb') as handle1:
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

# 2000+ taxis
sim_data_path = '/home/user/ClusterSetUp-All-to-One/%s-Data/' % CITY_NAME

#NUM_TAXIS = 300
#real_VANET_dataset_filename = 'No_Pandas_VANET_rome_august_edit_combined_3_days.pickle'
#real_positions_dataset_filename = 'No_Pandas_august_edit_taxi_positions_combined_3_days_rome.pickle' 

NUM_TAXIS = 2000
real_VANET_dataset_filename = 'Corrected_no_pandas_vanet_rome_29_days.pickle'
real_positions_dataset_filename = 'No_Pandas_august_edit_taxi_positions_combined_29_days_rome.pickle'


#output location
output_results_path = '/home/user/ABMTANET/maximal_connectedness/real_dataset_VANET_LCC_results/%s/%sish_taxis/' % (CITY_NAME, NUM_TAXIS)


# load the data in.
with open((sim_data_path+real_VANET_dataset_filename), 'rb') as handle0:
    vanet_dict = pickle.load(handle0)
handle0.close()

with open((sim_data_path+real_positions_dataset_filename), 'rb') as handle1:
    pos_dict = pickle.load(handle1)
handle1.close()


#SF geometries...
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

timestamp_list = sorted(list(vanet_dict.keys()))

for timestamp in timestamp_list[1000:1100]:
#for timestamp, values in vanet_dict.items():

    G = nx.Graph()
    edges_list = [] #so edge-y!
    #taxiAids_list = values['taxiAid']
    #taxiBids_list = values['taxiBid']

    edges_list = list(zip(vanet_dict[timestamp]['taxiAid'],vanet_dict[timestamp]['taxiBid']))

    G.add_edges_from(edges_list)
    LCC_list = list(max(nx.connected_components(G), key=len))
    num_taxis_in_LCC = len(LCC_list)

    LCC_latitudes_array = np.zeros(num_taxis_in_LCC)
    LCC_longitudes_array = np.zeros(num_taxis_in_LCC)

    lcc_taxi_counter = 0

    for taxi_id in LCC_list:
        if taxi_id in vanet_dict[timestamp]['taxiAid']:
            taxi_index = vanet_dict[timestamp]['taxiAid'].index(taxi_id)
            lon = vanet_dict[timestamp]['Alonglat'][taxi_index][0]
            lat = vanet_dict[timestamp]['Alonglat'][taxi_index][1]
#            a = values['Alonglat'][taxi_index]

        else:
            taxi_index = vanet_dict[timestamp]['taxiBid'].index(taxi_id)
            lon = vanet_dict[timestamp]['Blonglat'][taxi_index][0]
            lat = vanet_dict[timestamp]['Blonglat'][taxi_index][1]
#            b = values['Blonglat'][taxi_index]

        LCC_latitudes_array[lcc_taxi_counter] = lat
        LCC_longitudes_array[lcc_taxi_counter] = lon
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
    
    convex_hull_area_m2_list.append(lcc_hull.volume)
    convex_hull_perimeter_m_list.append(lcc_hull.area)

    #Sneaky Heatmap...
    all_taxis_longs, all_taxis_lats = list(zip(*pos_dict[timestamp]['longlat']))
    H2, xedges2, yedges2 = np.histogram2d(np.array(all_taxis_longs), np.array(all_taxis_lats), bins=(Xedges,Yedges))


    #PLotting....

    plt.figure()
    plt.imshow(np.rot90(H2), cmap=plt.cm.BuPu_r)
    plt.title('%s, num. Taxis:%i, time_step:%i, max_component: %i, total v2v exchanges: %i' % (CITY_NAME, NUM_TAXIS, timestamp, num_taxis_in_LCC, len(edges_list)))
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
    plt.savefig((output_results_path+('%s_real_dataset_%i_taxis_vanet_lcc_plot_TS_%i.png' % (CITY_NAME, NUM_TAXIS,timestamp))), dpi=300)
    #Overall VANET heatmap...
    #H, xedges2, yedges2 = np.histogram2d(

    print('timestamp= %i, LCC_area= %f LCC_perimeter= %f, P/A= %f' % (timestamp, lcc_hull.volume, lcc_hull.area, (lcc_hull.area/lcc_hull.volume)))















