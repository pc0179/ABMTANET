"""
life is raw.

15th Feb.


based on ABM_taxi_VANET_sim_feb13.py

the following code attempts to create a 'Null hypothesis" model...

namely randomly generated points rather than using taxi trajectories as taxi positions...


"""


#timer? IHKH
from timeit import default_timer as timer

# Import key libraries...
import numpy as np
import shapely #??? why is this needed so badly?
import pickle

# some useful Classes....
class TaxiAgent():

    def __init__(self,unique_id, starting_position):
        self.id = unique_id
        self.pos = starting_position
        self.old_pos_list = []
        self.t = 0 #personal timestep counter?


# Some Useful Functions:


def HaversineDistPC2(pos1,pos2):
    #where pos1 & pos2 are tuples: (longitude,latitude)
    lon1, lat1, lon2, lat2 = map(np.radians, [pos1[0],pos1[1],pos2[0], pos2[1]])
    dlon = lon2-lon1
    dlat = lat2-lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    Hdistance = 6371e3*c  #working in metres!
    return int(Hdistance)



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

def SelfHaversineDistMatrix(latitude_array, longitude_array):

    if latitude_array.size == longitude_array.size:
        MATRIX_LEN = latitude_array.size
        haversine_distance_mat = np.zeros((MATRIX_LEN,MATRIX_LEN))
    else:
        print("we have non-equal len arrays.... fawkward...")

    for i in range(0,MATRIX_LEN):
        all_other_lats = latitude_array[i:]
        all_other_longs = longitude_array[i:]
        current_lat_to_evaluate = np.ones_like(all_other_lats)*latitude_array[i]
        current_long_to_evaluate = np.ones_like(all_other_longs)*longitude_array[i]

        #dummy = VectorisedHaversineDistance(current_lat_to_evaluate, current_long_to_evaluate, all_other_lats, all_other_longs)
        
        haversine_distance_mat[i][i:] = VectorisedHaversineDistance(current_lat_to_evaluate, current_long_to_evaluate, all_other_lats, all_other_longs)

    return haversine_distance_mat

















#inputs for simulation...
#C225!

data_file_path =  '/home/toshiba/ABMTANET/gpickle_road_network_data/' 

"""
CITY_NAME = 'Roma'
SIM_RUN_DATE = 'NULL_2000taxis_0.5hr_v2vdata_15feb'

sensor_data_filename = 'roma_sensor_pos_dict_12feb.pickle' #'roma_sensor_pos_dict_21Jan.pickle'

# map boundaries....
MAX_LONG = 12.5387
MIN_LONG  = 12.4420
MAX_LAT = 41.928
MIN_LAT =  41.856

"""

#San Fran-phuckin-cisco
CITY_NAME = 'sf'
SIM_RUN_DATE = 'NULL_2000taxis_0.5hrs_v2vdata_14feb'
passenger_trip_data_filename = 'sf_passenger_trip_data_dict_28Jan.pickle'
sensor_data_filename = 'sf_sensor_pos_dict_28Jan.pickle'


# map boundaries....
MAX_LONG = -122.388
MIN_LONG = -122.479
MAX_LAT = 37.805
MIN_LAT = 37.733



"""
CITY_NAME = 'birmingham'
SIM_RUN_DATE = '2000taxis_0.5hrs_v2vdata_14feb'

passenger_trip_data_filename = 'birmingham_passenger_trip_data_dict_28Jan.pickle'
sensor_data_filename = 'birmingham_sensor_pos_dict_28Jan.pickle'
graph_data_filename = 'Highest_Protocol_Birmingham_central_Road_Network.gpickle'

"""



################# GENERAL SIMULATION PARAMS and Input DATA....
output_results_data_file_path =  '/home/toshiba/ABMTANET/simulation_results/'


model_space_width_m = HaversineDistPC2([MAX_LONG, MAX_LAT],[MIN_LONG, MAX_LAT])
model_space_height_m = HaversineDistPC2([MAX_LONG,MIN_LAT],[MAX_LONG, MAX_LAT])


# Some key model parameters:
LEN_SIM = int(60*60*0.5) #simulation length, in real seconds, 1s=1sim_Step

NUM_TAXIS = 2000
NUM_SENSORS = 500
NUM_TRIPS = 500 #500

V2V_MAXRANGE = 200 # metres
V2I_MAXRANGE = 100 # metres...
#MAX_ITERS = 600 # model iterations/real time simulated seconds in theory...
MAX_V = 20*(1000/3600) # m/s (equiv to 20km/h)


#Urban V2V LOS Model:
V2V_EXPO_MODEL_COEFFS = [-2.36945344, -0.08361897] #[0,1]# 
NORM_DIST_SCALING_FACTOR = 1/500 #max metres?
# Data Structures
# initiation loop, set out all taxi positions, trips, sensors... etc...
# we should only work with (long,lat)? avoid insane conversions? maybe...

"""
#To generate within script...
sensor_longitude_array = (np.random.rand(NUM_SENSORS,)*(MAX_LONG-MIN_LONG))+MIN_LONG
sensor_latitude_array = (np.random.rand(NUM_SENSORS,)*(MAX_LAT-MIN_LAT))+MIN_LAT

trip_longitude_array = (np.random.rand(NUM_TRIPS,)*(MAX_LONG-MIN_LONG))+MIN_LONG
trip_latitude_array = (np.random.rand(NUM_TRIPS,)*(MAX_LAT-MIN_LAT))+MIN_LAT

"""
#Otherwise load standard, 500 sensors from file:
with open((data_file_path + sensor_data_filename),'rb') as handle0:
    sensor_pos_dict = pickle.load(handle0)
handle0.close()

sensor_longitude_array = sensor_pos_dict['longitude']
sensor_latitude_array = sensor_pos_dict['latitude']


#Initial randomised taxi positions......
taxi_longitude_array = np.random.random(NUM_TAXIS)*(MAX_LONG-MIN_LONG) + MIN_LONG
taxi_latitude_array = np.random.random(NUM_TAXIS)*(MAX_LAT-MIN_LAT) + MIN_LAT
taxi_agent_list = []
for h in range(NUM_TAXIS):
    
    new_taxi_agent = TaxiAgent(h,[taxi_longitude_array[h], taxi_latitude_array[h]])

    taxi_agent_list.append(new_taxi_agent) 



taxi_message_set_dict = dict()
sensor_message_set_dict = dict()

for sensor_id in range(NUM_SENSORS):
    sensor_message_set_dict[sensor_id] = {sensor_id}
for taxi_id in range(NUM_TAXIS):
    taxi_message_set_dict[taxi_id]= set()



# Simulation Initiaion...

v2v_sharing_loc_dict = dict()
CIA_memory_dict= dict()
vanet_data_dict = dict()
vanet_node_loc_list_of_dicts = list()

################################ Simulation Begins NOW ~~~~~~~~~~~~~~~~~~~~~~~~#####

start_code_time = timer()


for time_step in range(0,LEN_SIM):

    print('simulation time step = %i' % time_step)

    # Distance Matricies evaluations...
    taxis_sensors_dist_mat =  HaversineDistMatrix(sensor_latitude_array, sensor_longitude_array, taxi_latitude_array, taxi_longitude_array)
    taxis_taxis_dist_mat = SelfHaversineDistMatrix(taxi_latitude_array, taxi_longitude_array)
#    taxis_trips_dist_mat = HaversineDistMatrix(start_passenger_trip_latitude_array, start_passenger_trip_longitude_array, taxi_latitude_array, taxi_longitude_array)

    nonzeroelements = np.nonzero(taxis_taxis_dist_mat)
    mean_dist = np.sum(taxis_taxis_dist_mat)/len(nonzeroelements[0])

    print('num nonzero elements: %i' % len(nonzeroelements[0]))
    print('mean distance between taxis: %f' % mean_dist)
    print('mean values of entire matrix: %f' % np.mean(taxis_taxis_dist_mat))

    #Comms message exchange
    #V2I - taxi-sensors
    taxis_sensors_within_range_index = np.where(taxis_sensors_dist_mat<=V2I_MAXRANGE)
    for i in range(0,taxis_sensors_within_range_index[0].size):

        sensor_id = taxis_sensors_within_range_index[0][i]
        taxi_id = taxis_sensors_within_range_index[1][i]

        #print('taxi %i exchanged with sensor %i at t=%i' % (taxi_id, sensor_id, time_step))
        current_taxi_message_set = taxi_message_set_dict[taxi_id]
        current_sensor_message_set = sensor_message_set_dict[sensor_id]
        combined_message_sets = current_taxi_message_set.union(current_sensor_message_set)

        taxi_message_set_dict[taxi_id] = combined_message_sets
        sensor_message_set_dict[sensor_id] = combined_message_sets



    #V2V - taxi-taxi
    taxis_taxis_within_range_index = np.where((taxis_taxis_dist_mat>0) & (taxis_taxis_dist_mat<=V2V_MAXRANGE))


    #V2V message transfer success probability model.. aka Urban VANET LOS Model...
    NUM_V2V_TAXIS = len(taxis_taxis_within_range_index[0])
    random_vars = np.random.rand(NUM_V2V_TAXIS)
    v2v_dists_array = taxis_taxis_dist_mat[taxis_taxis_within_range_index]

    P_V2V = np.exp(V2V_EXPO_MODEL_COEFFS[1])*np.exp(NORM_DIST_SCALING_FACTOR*V2V_EXPO_MODEL_COEFFS[0]*v2v_dists_array)

    v2v_test  = P_V2V - random_vars #r<P for successful V2V transmission
    v2v_exchange_success_index = np.where(v2v_test>0)


    # now randomise taxi-pair order of those within range and with "luck on their side"...
    shuffle_index = np.arange(taxis_taxis_within_range_index[0][v2v_exchange_success_index].size)
    np.random.shuffle(shuffle_index)

    print('len of taxis-taxis within V2V range... %i' % NUM_V2V_TAXIS)
    print('len of shuffle_index= %i' % (len(shuffle_index)))


    v2v_sharing_latitudes_array = np.zeros(len(shuffle_index))
    v2v_sharing_longitudes_array = np.zeros(len(shuffle_index))
    v2v_sharing_counter = 0

    v2v_edge_data_list = list()

    vanet_node_longitudes_list = []
    vanet_node_latitudes_list = []
    vanet_node_id_list = []

    for k in shuffle_index:

        Ataxi_id = taxis_taxis_within_range_index[0][k]
        Btaxi_id = taxis_taxis_within_range_index[1][k]

        v2v_edge_data_list.append(tuple([Ataxi_id, Btaxi_id]))
        #record V2V network... every timestep?

        Ataxi_message_set = taxi_message_set_dict[Ataxi_id]
        Btaxi_message_set = taxi_message_set_dict[Btaxi_id]

        if len(Ataxi_message_set)+len(Btaxi_message_set)>0:
            combined_taxis_message_sets = Ataxi_message_set.union(Btaxi_message_set)

            taxi_message_set_dict[Ataxi_id] = combined_taxis_message_sets
            taxi_message_set_dict[Btaxi_id] = combined_taxis_message_sets
            #print('taxiA %i exchanged with taxiB %i at t=%i' % (Ataxi_id, Btaxi_id, time_step))
            #print(combined_taxis_message_sets)

            #RECORD mid-points of V2V sharing to generate 'heat-map'?
            v2v_sharing_latitudes_array[v2v_sharing_counter] = (taxi_latitude_array[Ataxi_id]+taxi_latitude_array[Btaxi_id])/2
            v2v_sharing_longitudes_array[v2v_sharing_counter] = (taxi_longitude_array[Ataxi_id]+taxi_longitude_array[Btaxi_id])/2
            v2v_sharing_counter +=1

            vanet_node_latitudes_list.append(taxi_latitude_array[Ataxi_id])
            vanet_node_latitudes_list.append(taxi_latitude_array[Btaxi_id])

            vanet_node_longitudes_list.append(taxi_longitude_array[Ataxi_id])
            vanet_node_latitudes_list.append(taxi_longitude_array[Btaxi_id])

            vanet_node_id_list.append(Ataxi_id)
            vanet_node_id_list.append(Btaxi_id)

    #more thorough V2V data analysis...
    vanet_data_dict[time_step] = {'edge_list':v2v_edge_data_list, 'latitude':v2v_sharing_latitudes_array, 'longitude':v2v_sharing_longitudes_array}

    vanet_node_loc_list_of_dicts.append({})

    for taxi_node_id in taxis_taxis_within_range_index[0]:
        if taxi_node_id not in vanet_node_loc_list_of_dicts[time_step]:
            vanet_node_loc_list_of_dicts[time_step][taxi_node_id] = tuple([taxi_longitude_array[taxi_node_id], taxi_latitude_array[taxi_node_id]]) 

    for taxi_node_id in taxis_taxis_within_range_index[1]:
        if taxi_node_id not in vanet_node_loc_list_of_dicts[time_step]:
            vanet_node_loc_list_of_dicts[time_step][taxi_node_id] = tuple([taxi_longitude_array[taxi_node_id], taxi_latitude_array[taxi_node_id]])









    #Let the games begin.

    #UPDATE ALL TAXIS POSITIONS

    taxi_longitude_array = np.random.random(NUM_TAXIS)*(MAX_LONG-MIN_LONG) + MIN_LONG
    taxi_latitude_array = np.random.random(NUM_TAXIS)*(MAX_LAT-MIN_LAT) + MIN_LAT

    stupid_taxi_counter = 0
    for taxi in taxi_agent_list:

        #taxi.t +=1
        taxi.t = time_step
        taxi.pos = [taxi_longitude_array[stupid_taxi_counter], taxi_latitude_array[stupid_taxi_counter]]
        stupid_taxi_counter +=1


    CIA_memory_dict[time_step] = {'lon':taxi_longitude_array, 'lat':taxi_latitude_array}




end_code_time = timer()

print('%i iterations in: ' % LEN_SIM)
print(end_code_time-start_code_time)




with open((output_results_data_file_path +('%s_cluster_taxi_messages_results_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle:
    pickle.dump(taxi_message_set_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open((output_results_data_file_path +('%s_sensor_messages_results_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle2:
    pickle.dump(sensor_message_set_dict, handle2, protocol = pickle.HIGHEST_PROTOCOL)



#sensor_locs_dict = {'lon':sensor_longitude_array, 'lat':sensor_latitude_array}
#with open((output_results_data_file_path +('%s_sensor_locations_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle3:
#    pickle.dump(sensor_locs_dict, handle3, protocol = pickle.HIGHEST_PROTOCOL)

#with open((output_results_data_file_path +('%s_link_traffic_count_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle3:
#    pickle.dump(link_dict, handle3, protocol = pickle.HIGHEST_PROTOCOL)


#with open((output_results_data_file_path +('%s_passenger_trip_results_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle4:
#    pickle.dump(passenger_trip_results_dict, handle4, protocol = pickle.HIGHEST_PROTOCOL)
#handle4.close()

#save general model details to make plotting easier?
#general_model_params_dict = {'city':CITY_NAME,'model_width':model_space_width_m, 'model_height':model_space_height_m, 'sim_len':LEN_SIM, 'num_taxis':NUM_TAXIS, 'num_sensors':NUM_SENSORS, 'num_trips':NUM_TRIPS, 'max_v':MAX_V, 'min_lat': MIN_LAT, 'max_lat':MAX_LAT, 'min_lon':MIN_LONG,'max_lon':MAX_LONG}

#with open((output_results_data_file_path +('%s_general_model_params_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle5:
#    pickle.dump(general_model_params_dict, handle5, protocol = pickle.HIGHEST_PROTOCOL)
#handle5.close()


"""
# save taxi data stored in class variables...

taxi_hist_pos_dict = dict()

for taxi in taxi_agent_list:

    taxi_hist_pos_dict[taxi.id] = taxi.old_pos_list
 
with open((output_results_data_file_path +('%s_taxi_agent_position_data_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle6:
    pickle.dump(taxi_hist_pos_dict, handle6, protocol = pickle.HIGHEST_PROTOCOL)
handle6.close()
"""

#with open((output_results_data_file_path +('%s_taxi_route_break_dict_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle5:
#    pickle.dump(taxi_route_break_dict, handle5, protocol = pickle.HIGHEST_PROTOCOL)

### save trip start points...
#passenger_trip_start_locs_dict = {'latitude': start_passenger_trip_latitude_array, 'longitude':start_passenger_trip_longitude_array}

#with open((output_results_data_file_path+('%s_passenger_trip_start_locs_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle7:
#    pickle.dump(passenger_trip_start_locs_dict, handle7, protocol = pickle.HIGHEST_PROTOCOL)
#handle7.close()

#with open((output_results_data_file_path +'%s_v2v_sharing_loc_heatmap_dict_%s.pickle' %(CITY_NAME, SIM_RUN_DATE)), 'wb') as handle8:
#    pickle.dump( v2v_sharing_loc_dict, handle8, protocol=pickle.HIGHEST_PROTOCOL)
#handle8.close()

# v2v edge data....


with open((output_results_data_file_path +'%s_vanet_data_dict_%s.pickle' %(CITY_NAME, SIM_RUN_DATE)), 'wb') as handle9:
    pickle.dump(vanet_data_dict, handle9, protocol=pickle.HIGHEST_PROTOCOL)
handle9.close()



with open((output_results_data_file_path +'%s_vanet_node_loc_dict_%s.pickle' %(CITY_NAME, SIM_RUN_DATE)), 'wb') as handle10:
    pickle.dump(vanet_node_loc_list_of_dicts, handle10, protocol=pickle.HIGHEST_PROTOCOL)
handle10.close()




"""


PDR = []
for key, values in sensor_message_set_dict.items():

    PDR.append(len(values))


import matplotlib.pyplot as plt

plt.figure()
plt.plot(PDR, 'ok')
plt.show()



PRR = []
for key, values in taxi_message_set_dict.items():

    PRR.append(len(values))


import matplotlib.pyplot as plt

plt.figure()
plt.plot(PRR, 'ok')
plt.show()

"""


























       
































