"""
a more 'road-link' orientated passenger trip and sensor position generator script for ABMTANET simulations....
21st January 2k19
phuuuck.
"""



import pickle
import numpy as np
from networkx import read_gpickle, shortest_path, has_path


###### OSM FILTERED ROAD MAP DATAFILE...

data_file_path =  '/home/user/ABMTANET/gpickle_road_network_data/'
road_network_filename = 'cologne_central_8km2.gpickle' #'Highest_Protocol_Roma_Centrale_Road_Network.gpickle'
CITY_NAME = 'koln' #'roma'
SIM_RUN_DATE = '21jan' #'18Jan'

# road network data
road_network = read_gpickle(data_file_path + road_network_filename)
node_id_list = list(road_network.nodes()) 
node_longitude_dict = dict(road_network.nodes(data='x'))
node_latitude_dict = dict(road_network.nodes(data='y'))

MAX_LONG = max(node_longitude_dict.values())
MIN_LONG  = min(node_longitude_dict.values())
MAX_LAT = max(node_latitude_dict.values())
MIN_LAT =  min(node_latitude_dict.values())


node_id_list = list(road_network.nodes()) 
edge_list = list(road_network.edges())


#### SENSOR LOCATIONS ~~~~~
NUM_SENSORS = 50


sensor_link_index = np.random.choice(len(edge_list),NUM_SENSORS)

link_dict = dict()
#sensor_longitude_dict = dict()
#sensor_latitude_dict = dict()


for j in range(0,len(edge_list)):

    link_dict[(edge_list[j][0],  edge_list[j][1])] = {'len':road_network[edge_list[j][0]][edge_list[j][1]][0]['length'], 'traffic_count':0}




SENSOR_ID = 0
sensorID_link_dict = dict()
sensor_longitude_list = list()
sensor_latitude_list = list()

for s in sensor_link_index:

    link_dict[(edge_list[s][0],  edge_list[s][1])] = {'len':road_network[edge_list[s][0]][edge_list[s][1]][0]['length'], 'traffic_count':0, 'sensor_id':SENSOR_ID}

    pointA = np.array([node_longitude_dict[edge_list[s][0]], node_latitude_dict[edge_list[s][0]]])
    pointB = np.array([node_longitude_dict[edge_list[s][1]], node_latitude_dict[edge_list[s][1]]])
    link_mid_point = list((pointA+pointB)/2)

#    sensor_longitude_dict[s] = link_mid_point[0]
#    sensor_latitude_dict[s] = link_mid_point[1]

    sensor_longitude_list.append(link_mid_point[0])
    sensor_latitude_list.append(link_mid_point[1])
    sensorID_link_dict[s] = (edge_list[j][0],  edge_list[j][1])

    SENSOR_ID +=1


sensor_longitude_array = (np.random.rand(NUM_SENSORS,)*(MAX_LONG-MIN_LONG))+MIN_LONG
sensor_latitude_array = (np.random.rand(NUM_SENSORS,)*(MAX_LAT-MIN_LAT))+MIN_LAT

sensor_pos_dict = {'longitude':sensor_longitude_array, 'latitude':sensor_latitude_array}

with open((data_file_path+('%s_sensor_pos_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle1:
    pickle.dump(sensor_pos_dict, handle1, protocol = pickle.HIGHEST_PROTOCOL)




##### PASSENGER TRIPS ########

NUM_TRIPS = 100

passenger_trip_dict = dict()
start_passenger_trip_longitude_array = np.zeros(NUM_TRIPS)
start_passenger_trip_latitude_array = np.zeros(NUM_TRIPS)

for passenger_trip_id in range(NUM_TRIPS):
    
    trip_start_node_id = np.random.choice(node_id_list)
    trip_target_node_id = np.random.choice(node_id_list)
    while trip_start_node_id == trip_target_node_id:
        trip_target_node_id = np.random.choice(node_id_list)

    trip_has_path = has_path(road_network, trip_start_node_id, trip_target_node_id)
    while trip_has_path is False:
        print('we have a dud path, TRIP id: %i, nodes: %i, %i' % (passenger_trip_id, trip_start_node_id, trip_target_node_id))
        trip_target_node_id = np.random.choice(node_id_list)
        trip_start_node_id = np.random.choice(node_id_list)
        trip_has_path = has_path(road_network, trip_start_node_id, trip_target_node_id)

    if trip_has_path is True:

        passenger_trip_waypoints = shortest_path(road_network, trip_start_node_id, trip_target_node_id, weight='length')

        passenger_trip_start_pos = [node_longitude_dict[trip_start_node_id],node_latitude_dict[trip_start_node_id]]
        passenger_trip_destination_pos = [node_longitude_dict[trip_target_node_id],node_latitude_dict[trip_target_node_id]]
        
        start_passenger_trip_longitude_array[passenger_trip_id] = node_longitude_dict[trip_start_node_id]
        start_passenger_trip_latitude_array[passenger_trip_id] = node_latitude_dict[trip_start_node_id]

        passenger_trip_dict[passenger_trip_id] = {'start_pos':passenger_trip_start_pos, 'start_node':trip_start_node_id, 'dest_pos':passenger_trip_destination_pos, 'dest_node':trip_target_node_id, 'route':passenger_trip_waypoints, 'route_len':len(passenger_trip_waypoints)}

### save passenger trip data
with open((data_file_path+('%s_passenger_trip_data_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle:
    pickle.dump(passenger_trip_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

