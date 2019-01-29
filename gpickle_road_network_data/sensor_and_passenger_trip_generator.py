"""

getting to that stage where generating trips and sensor locations is tedious and slowing down debugging/progress.

aim to standardise this, have pre-set 'pickles' of trips and sensors for any osmnx filtered OSM road map

18.01.2k19

"""

import pickle
import numpy as np
from networkx import read_gpickle, shortest_path, has_path

###### OSM FILTERED ROAD MAP DATAFILE...
"""
#on c225
data_file_path =  '/home/toshiba/MiniTaxiFleets/gpickle_road_network_data/'
road_network_filename = 'Highest_Protocol_Roma_Centrale_Road_Network.gpickle'
CITY_NAME = 'roma'
SIM_RUN_DATE = '18Jan'
"""

#on C207
data_file_path =  '/home/user/ABMTANET/gpickle_road_network_data/'
#road_network_filename = 'Highest_Protocol_SF_Central_Road_Network.gpickle'
road_network_filename = 'Highest_Protocol_Birmingham_central_Road_Network.gpickle'
CITY_NAME = 'birmingham'
SIM_RUN_DATE = '28Jan'

# road network data
road_network = read_gpickle(data_file_path + road_network_filename)
node_id_list = list(road_network.nodes()) 
node_longitude_dict = dict(road_network.nodes(data='x'))
node_latitude_dict = dict(road_network.nodes(data='y'))

MAX_LONG = max(node_longitude_dict.values())
MIN_LONG  = min(node_longitude_dict.values())
MAX_LAT = max(node_latitude_dict.values())
MIN_LAT =  min(node_latitude_dict.values())

#### SENSOR LOCATIONS ~~~~~
NUM_SENSORS = 500
sensor_longitude_array = (np.random.rand(NUM_SENSORS,)*(MAX_LONG-MIN_LONG))+MIN_LONG
sensor_latitude_array = (np.random.rand(NUM_SENSORS,)*(MAX_LAT-MIN_LAT))+MIN_LAT

sensor_pos_dict = {'longitude':sensor_longitude_array, 'latitude':sensor_latitude_array}

with open((data_file_path+('%s_sensor_pos_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle1:
    pickle.dump(sensor_pos_dict, handle1, protocol = pickle.HIGHEST_PROTOCOL)


##### PASSENGER TRIPS ########

NUM_TRIPS = 4000

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



        #new_passenger_trip = PassengerTrip(passenger_trip_id, passenger_trip_start_pos, trip_start_node_id, passenger_trip_destination_pos, trip_target_node_id, passenger_trip_waypoints)

#        passenger_trip_list.append(new_passenger_trip)



