"""

as of 11march2k19 bitch.

it has been decided that lower values of alpha 0-1/3 of great interest.

new folder set up to deal with this, once again re-run some simulations
for both rome and Sf

SF is currently being dealt with in this script

alpha = [0,0.05,0.1,0.15,0.2,0.25,0.3]


"""


import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()



def VectorisedHaversineDistance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    #Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    #Implementing Haversine Formula: 
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),  
                          np.multiply(np.cos(lat1), 
                                      np.multiply(np.cos(lat2), 
                                                  np.power(np.sin(np.divide(dlon, 2)), 2))))
    c = np.multiply(2, np.arcsin(np.sqrt(a)))
    r = 6371000 #always in metres!

    return c*r



def HaversineDistMatrix(Alatitude_array, Alongitude_array, Blatitude_array, Blongitude_array):
    """ 
    first set of arrays (A) will end up as the column of the dist mat
    whereas second set, (B) will end up as the rows...
    """
    if Alatitude_array.size == Alongitude_array.size:
        MAT_ROW_LEN = Alatitude_array.size
    else:
        print('first array issues... not equal lengths')

    if Blatitude_array.size == Blongitude_array.size:
        MAT_COL_LEN = Blatitude_array.size
    else:
        print('second array issues... not equal lengths')
    #print('cols: %i, rows: %i' % (MAT_COL_LEN, MAT_ROW_LEN))
    haversine_dist_mat = np.zeros((MAT_ROW_LEN,MAT_COL_LEN))

    for i in range(0,MAT_ROW_LEN):

        Fx = np.ones_like(Blongitude_array)*Alongitude_array[i]
        Fy = np.ones_like(Blatitude_array)*Alatitude_array[i]
        haversine_dist_mat[i] = VectorisedHaversineDistance(Blatitude_array,Blongitude_array,Fy,Fx)

    return haversine_dist_mat




#alpha_list = [0,1/3,0.5,2/3,1]


alpha_list = [0,0.05,0.1,0.15,0.2,0.25,0.3]
CITY_NAME = 'SF'
SIM_RUN_DATE = '11March_C225'



data_file_path =  '/home/toshiba/Dropbox/ABMTANET_working_dbx/low_alpha_values/%s/' % CITY_NAME

output_data_file_path =  '/home/toshiba/Dropbox/ABMTANET_working_dbx/low_alpha_values/%s/sim_input_datasets/' % CITY_NAME

road_network_filename = 'Highest_Protocol_SF_Central_Road_Network.gpickle'



# San Francisco map boundaries....
MAX_LONG = -122.388
MIN_LONG = -122.479
MAX_LAT = 37.805
MIN_LAT = 37.733

# road network data
road_network = nx.read_gpickle(data_file_path + road_network_filename)
node_id_list = list(road_network.nodes()) 
node_longitude_dict = dict(road_network.nodes(data='x'))
node_latitude_dict = dict(road_network.nodes(data='y'))



node_id_list = list(road_network.nodes()) 
edge_list = list(road_network.edges())



NUM_SENSORS = 1000
V2I_MAX_RANGE = 100 #metres...
NUM_TRIPS = 4000

#### SENSOR LOCATIONS ~~~~~~~~~~~~#~##~#####~~~#
sensor_longitude_array = np.random.random(NUM_SENSORS)*(MAX_LONG-MIN_LONG) + MIN_LONG
sensor_latitude_array = np.random.random(NUM_SENSORS)*(MAX_LAT-MIN_LAT) + MIN_LAT

BIG_LINK_DICT = dict()
edge_index_dict = dict()
edge_midpoint_lat_array = np.zeros(len(edge_list))
edge_midpoint_long_array = np.zeros(len(edge_list))

for g in range(0,len(edge_list)):

    #Record Sensor ID?
    BIG_LINK_DICT[(edge_list[g][0],  edge_list[g][1])] = {'len':road_network[edge_list[g][0]][edge_list[g][1]][0]['length'], 'traffic_count':0, 'sensor_id':[]}

    edge_index_dict[g] = (edge_list[g][0], edge_list[g][1])
    lat_midpoint = np.mean([node_latitude_dict[edge_list[g][0]], node_latitude_dict[edge_list[g][1]]])
    long_midpoint = np.mean([node_longitude_dict[edge_list[g][0]], node_longitude_dict[edge_list[
g][1]]]) 
    edge_midpoint_lat_array[g] = lat_midpoint
    edge_midpoint_long_array[g] = long_midpoint

sensor_pos_dict = {'longitude':sensor_longitude_array, 'latitude':sensor_latitude_array}


# save sensor positions and link/edge index dictionary
with open((output_data_file_path+('%s_sensor_pos_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle1:
    pickle.dump(sensor_pos_dict, handle1, protocol = pickle.HIGHEST_PROTOCOL)
handle1.close()

with open((output_data_file_path+('%s_edge_index_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle2:
    pickle.dump(edge_index_dict, handle2, protocol = pickle.HIGHEST_PROTOCOL)
handle2.close()



# distance matrix, sensor locations and edge mid points
sensor_edge_dist_matrix = HaversineDistMatrix(edge_midpoint_lat_array, edge_midpoint_long_array, sensor_latitude_array, sensor_longitude_array)

# update BIG_LINK_DICT with sensors on links/edges...
nearest_sensor_to_edge_index = np.argmin(sensor_edge_dist_matrix, axis=0)

for h in range(len(nearest_sensor_to_edge_index)):
    nodeAid = edge_index_dict[nearest_sensor_to_edge_index[h]][0]
    nodeBid = edge_index_dict[nearest_sensor_to_edge_index[h]][1]
    BIG_LINK_DICT[(nodeAid, nodeBid)]['sensor_id'].append(h)



### generate useful networks...
road_network2 = nx.Graph()

for i in range(len(alpha_list)):

    weight_type = 'type'+str(i)
    print(weight_type)

    for edge_id, values in edge_index_dict.items():

        link_length = BIG_LINK_DICT[(values[0],values[1])]['len']
        dist_to_nearest_sensor = np.min(sensor_edge_dist_matrix[edge_id][:])
        link_weight = ((1-alpha_list[i])*link_length + (alpha_list[i]*dist_to_nearest_sensor))/V2I_MAX_RANGE

        road_network2.add_weighted_edges_from([(values[0],values[1],link_weight)], weight= weight_type)



##### PASSENGER TRIPS ########
def PassengerODGenerator(Road_Graph, number_of_trips, list_of_node_ids):

    output_OD_array = np.zeros((number_of_trips,2))

    for passenger_trip_id in range(number_of_trips):
        
        print(passenger_trip_id)

        trip_start_node_id = np.random.choice(list_of_node_ids)
        trip_target_node_id = np.random.choice(list_of_node_ids)

        while trip_start_node_id == trip_target_node_id:
            trip_target_node_id = np.random.choice(list_of_node_ids)

        trip_has_path = nx.has_path(G=Road_Graph, source=trip_start_node_id, target=trip_target_node_id)
        while trip_has_path is False:
            print('we have a dud path, TRIP id: %i, nodes: %i, %i' % (passenger_trip_id, trip_start_node_id, trip_target_node_id))
            trip_target_node_id = np.random.choice(list_of_node_ids)
            trip_start_node_id = np.random.choice(list_of_node_ids)
            trip_has_path = nx.has_path(G=Road_Graph, source=trip_start_node_id, target=trip_target_node_id)

        if trip_has_path is True:

            output_OD_array[passenger_trip_id][0] = trip_start_node_id
            output_OD_array[passenger_trip_id][1] = trip_target_node_id



    return output_OD_array


def PassengerRouteGenerator(Road_Graph, passenger_OD_array, weight_type, node_longtiude_dict, node_latitude_dict):

    start_passenger_trip_longitude_array = np.zeros(len(passenger_OD_array))
    start_passenger_trip_latitude_array = np.zeros(len(passenger_OD_array))

    output_passenger_trip_dict = dict()

    for passenger_trip_id in range(len(passenger_OD_array)):
        
        print(passenger_trip_id)

        trip_start_node_id = passenger_OD_array[passenger_trip_id][0]
        trip_target_node_id = passenger_OD_array[passenger_trip_id][1]

        passenger_trip_start_pos = [node_longitude_dict[trip_start_node_id],node_latitude_dict[trip_start_node_id]]
        passenger_trip_destination_pos = [node_longitude_dict[trip_target_node_id],node_latitude_dict[trip_target_node_id]]
        
        start_passenger_trip_longitude_array[passenger_trip_id] = node_longitude_dict[trip_start_node_id]
        start_passenger_trip_latitude_array[passenger_trip_id] = node_latitude_dict[trip_start_node_id]

        # routing using standard dijkstra...................
        shortest_route_waypoints = nx.shortest_path(G=road_network2,source=trip_start_node_id, target=trip_target_node_id, method='dijkstra',weight=weight_type) 

        output_passenger_trip_dict[passenger_trip_id] = {'start_pos':passenger_trip_start_pos, 'start_node':trip_start_node_id, 'dest_pos':passenger_trip_destination_pos, 'dest_node':trip_target_node_id, 'route': shortest_route_waypoints}

    return output_passenger_trip_dict





passenger_trip_node_array = PassengerODGenerator(road_network2, NUM_TRIPS, node_id_list)


all_types_passenger_trip_dict = dict()
all_types_passenger_trip_dict['type0'] = PassengerRouteGenerator(road_network2, passenger_trip_node_array, 'type0', node_longitude_dict, node_latitude_dict)
all_types_passenger_trip_dict['type1'] = PassengerRouteGenerator(road_network2, passenger_trip_node_array, 'type1', node_longitude_dict, node_latitude_dict)
all_types_passenger_trip_dict['type2'] = PassengerRouteGenerator(road_network2, passenger_trip_node_array, 'type2', node_longitude_dict, node_latitude_dict)
all_types_passenger_trip_dict['type3'] = PassengerRouteGenerator(road_network2, passenger_trip_node_array, 'type3', node_longitude_dict, node_latitude_dict)
all_types_passenger_trip_dict['type4'] = PassengerRouteGenerator(road_network2, passenger_trip_node_array, 'type4', node_longitude_dict, node_latitude_dict)

all_types_passenger_trip_dict['type5'] = PassengerRouteGenerator(road_network2, passenger_trip_node_array, 'type5', node_longitude_dict, node_latitude_dict)
all_types_passenger_trip_dict['type6'] = PassengerRouteGenerator(road_network2, passenger_trip_node_array, 'type6', node_longitude_dict, node_latitude_dict)



# EVALUATION?ANALSYIS>?~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

all_eval_metrics_dict = dict()

for weight_type, values in all_types_passenger_trip_dict.items():

    route_length_list = []
    hop_count_list = []
    sensor_count_list = []
    trip_id_list = [] 

    for trip_id, trip_values in values.items():

        route_link_len_list = []
        route_sensor_count = 0
        route_hop_count = len(trip_values['route'])
        waypoints = trip_values['route']

# there appears to be some issues regards directionality of links... (nodeAid, nodeBid) not the same as (nodeBid, nodeAid).....????????????????????????????????
        for j in range(route_hop_count-1):
            if (waypoints[j],waypoints[j+1]) in BIG_LINK_DICT:
                route_link_len_list.append(BIG_LINK_DICT[(waypoints[j],waypoints[j+1])]['len'])

                if len(BIG_LINK_DICT[(waypoints[j],waypoints[j+1])]['sensor_id'])>0:
                    route_sensor_count +=1
            else:
                route_link_len_list.append(BIG_LINK_DICT[(waypoints[j+1],waypoints[j])]['len'])
                print('hmmmm..... %i' % trip_id)
                
                if len(BIG_LINK_DICT[(waypoints[j+1],waypoints[j])]['sensor_id'])>0:
                    route_sensor_count +=1
              
        #route_eval_dict['type1'][key] = {'len':sum(route_length_list),'hop_count':route_hop_count, 'sensor_count':route_sensor_count}
        route_length_list.append(sum(route_link_len_list))
        hop_count_list.append(route_hop_count)
        sensor_count_list.append(route_sensor_count)
        trip_id_list.append(trip_id)


    all_eval_metrics_dict[weight_type] = {'trip_id': trip_id_list, 'hop_count': hop_count_list , 'sensor_count': sensor_count_list , 'trip_len': route_length_list}



###### plotting needs to be done in a loop by type.... allllloooooowwwww.......#########################################

from matplotlib.pyplot import cm

plotting_colours_iter = iter(cm.rainbow(np.linspace(0,1,len(all_eval_metrics_dict.keys()))))
plotting_legend = []
trip_length_bins = np.linspace(0,18000,37)
plotting_trip_length_mid_points = trip_length_bins[1:] - (trip_length_bins[1:]-trip_length_bins[0:-1])/2
route_length_histo_dict = dict()

plt.figure()
for weight_type, values in all_eval_metrics_dict.items():

    trip_len_hist, trip_binedges = np.histogram(values['trip_len'], trip_length_bins)

    route_length_histo_dict[weight_type] = trip_len_hist/NUM_TRIPS

    new_plot_colour = next(plotting_colours_iter)

    plt.plot(plotting_trip_length_mid_points,route_length_histo_dict[weight_type], c=new_plot_colour)    

    plotting_legend.append(weight_type)



plt.xlabel('Trip Length/[m]')
plt.ylabel('Frequency')
plt.legend(plotting_legend)
plt.title('Normalised Trip Length Histo. for diff. SF graph weights (Num. Trips:%i)' % NUM_TRIPS)
plt.savefig('alpha_Trip_len_histo_normalised_SF.png',dpi=400, pad_inches=0.05)
plt.show()




# sensor count distribution.....
sensor_count_bins = np.linspace(0,50,51)
plotting_sensor_count_mid_points = sensor_count_bins[1:] - (sensor_count_bins[1:]-sensor_count_bins[0:-1])/2
plotting_colours_iter2 = iter(cm.rainbow(np.linspace(0,1,len(all_eval_metrics_dict.keys()))))
plotting_legend2 = []
sensor_count_histo_dict = dict()

plt.figure()
for weight_type, values in all_eval_metrics_dict.items():

    sensor_count_hist, sensor_binedges = np.histogram(values['sensor_count'], sensor_count_bins)

    sensor_count_histo_dict[weight_type] = sensor_count_hist/NUM_TRIPS

    new_plot_colour2 = next(plotting_colours_iter2)

    plt.plot(plotting_sensor_count_mid_points,sensor_count_histo_dict[weight_type], c=new_plot_colour2)    

    plotting_legend2.append(weight_type)


plt.xlabel('Sensor Count')
plt.ylabel('Frequency')
plt.legend(plotting_legend2)
plt.title('Normalised Trip Sensor Count Histo. for diff. SF graph weights (num. sensors:%i)' % NUM_SENSORS)
plt.xlim([0,30])
plt.savefig('alpha_sensor_count_histo.png',dpi=400, pad_inches=0.05)
plt.show()






with open((output_data_file_path +('low_ALPHA_param_passenger_trip_dict_%s_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle30:
    pickle.dump(all_types_passenger_trip_dict, handle30, protocol=pickle.HIGHEST_PROTOCOL)
handle30.close()

with open((output_data_file_path +('low_ALPHA_param_link_dict_%s_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle31:
    pickle.dump(BIG_LINK_DICT, handle31, protocol=pickle.HIGHEST_PROTOCOL)
handle31.close()

with open((output_data_file_path +('low_ALPHA_node_longitude_dict_%s_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle32:
    pickle.dump(node_longitude_dict, handle32, protocol=pickle.HIGHEST_PROTOCOL)
handle32.close()

with open((output_data_file_path +('low_ALPHA_weights_node_latitude_dict_%s_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle33:
    pickle.dump(node_latitude_dict, handle33, protocol=pickle.HIGHEST_PROTOCOL)
handle33.close()


with open((output_data_file_path +('low_ALPHA_eval_metrics_dict_%s_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle34:
    pickle.dump(all_eval_metrics_dict, handle34, protocol=pickle.HIGHEST_PROTOCOL)
handle34.close()



nx.write_gpickle(road_network2, "LOW_Alpha_SF_Graph.gpickle")










"""
#### plotting some results/evaluation between routing strategies....


import matplotlib.pyplot as plt
plt.ion()

plt.figure()
plt.plot(type0_route_length_list, type0_sensor_count_list,'ok')
plt.plot(type1_route_length_list, type1_sensor_count_list, '+r')
plt.xlabel('Trip Length/[m]')
plt.ylabel('Trip Sensor Count')
plt.legend(['Type0','type1'])
plt.title('Trip length vs sensor count for different SF graph weights')
plt.savefig('Trip_len_VS_sensor_count.png',dpi=400, pad_inches=0.05)
plt.show()


type0_route_length_array = np.array(type0_route_length_list)
type1_route_length_array = np.array(type1_route_length_list)

type0_sensor_count_array = np.array(type0_sensor_count_list)
type1_sensor_count_array = np.array(type1_sensor_count_list)

diff_route_length_array = type1_route_length_array - type0_route_length_array
diff_sensor_count_array = type1_sensor_count_array - type0_sensor_count_array

plt.figure()
plt.plot(diff_route_length_array, diff_sensor_count_array,'*b')
plt.xlabel('DIFF Trip Length/[m]')
plt.ylabel('DIFF Trip Sensor Count')
plt.legend(['Type0','type1'])
plt.title('Type1-Type0 results')
plt.savefig('DIFF_type1_0_results.png',dpi=400, pad_inches=0.05)
plt.show()




# trip length distributions....
trip_length_bins = np.linspace(0,18000,37)
plotting_trip_length_mid_points = trip_length_bins[1:] - (trip_length_bins[1:]-trip_length_bins[0:-1])/2

type0_trip_len_hist, trip_binedges = np.histogram(type0_route_length_list, trip_length_bins)
type1_trip_len_hist, trip_binedges = np.histogram(type1_route_length_list, trip_length_bins)

plt.figure()
plt.plot(plotting_trip_length_mid_points, type0_trip_len_hist, '-ok')
plt.plot(plotting_trip_length_mid_points, type1_trip_len_hist, '-+r')
plt.xlabel('Trip Length/[m]')
plt.ylabel('Frequency Count')
plt.legend(['Type0','type1'])
plt.title('Trip Length Histo. for diff. SF graph weights')
plt.show()

    #NORMALISED trip length Distirbution...
plt.figure()
plt.plot(plotting_trip_length_mid_points, type0_trip_len_hist/NUM_TRIPS, '-ok')
plt.plot(plotting_trip_length_mid_points, type1_trip_len_hist/NUM_TRIPS, '-+r')
plt.xlabel('Trip Length/[m]')
plt.ylabel('Frequency')
plt.legend(['Type0','type1'])
plt.title('Normalised Trip Length Histo. for diff. SF graph weights (Num. Trips:%i)' % NUM_TRIPS)
plt.savefig('Trip_len_histo_normalised.png',dpi=400, pad_inches=0.05)
plt.show()




# sensor count distribution.....
sensor_count_bins = np.linspace(0,50,26)
plotting_sensor_count_mid_points = sensor_count_bins[1:] - (sensor_count_bins[1:]-sensor_count_bins[0:-1])/2

type0_sensor_count_hist, sensor_binedges = np.histogram(type0_sensor_count_list,sensor_count_bins)
type1_sensor_count_hist, sensor_binedges = np.histogram(type1_sensor_count_list,sensor_count_bins)
plt.figure()
plt.plot(plotting_sensor_count_mid_points, type0_sensor_count_hist, '-ok')
plt.plot(plotting_sensor_count_mid_points, type1_sensor_count_hist, '-+r')
plt.xlabel('Sensor Count')
plt.ylabel('Frequency')
plt.legend(['Type0','type1'])
plt.title('Trip Sensor Count Histo. for diff. SF graph weights')
plt.xlim([0,30])
plt.show()

    # NORMALISED sensor count distribution
plt.figure()
plt.plot(plotting_sensor_count_mid_points, type0_sensor_count_hist/NUM_SENSORS, '-ok')
plt.plot(plotting_sensor_count_mid_points, type1_sensor_count_hist/NUM_SENSORS, '-+r')
plt.xlabel('Sensor Count')
plt.ylabel('Frequency')
plt.legend(['Type0','type1'])
plt.title('Trip Sensor Count Histo. for diff. SF graph weights (num. sensors:%i)' % NUM_SENSORS)
plt.xlim([0,30])
plt.savefig('sensor_count_histo.png',dpi=400, pad_inches=0.05)
plt.show()






"""




