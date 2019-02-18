"""
life is raw
attempting a re(^?)-write... of ABM taxi VANET simulation code...
Feb 13th 2k19

one key aspect of this is to avoid generating trips within the code...
really should aim to have a library/list of pre-approved/checked trips (that have a path!)
and the taxis should pick from there when they are desperate...

also need to ensure these phuckers are actually moving... allow

"""


#timer? IHKH
from timeit import default_timer as timer

# Import key libraries...
import numpy as np
import shapely #??? why is this needed so badly?
from networkx import has_path, shortest_path, read_gpickle
import pickle

# some useful Classes....
class TaxiAgent():

    def __init__(self,unique_id, starting_position, start_node_id, trip_destination_position, trip_destination_node_id, trip_waypoints, next_node_enroute_pos, first_link_exit_time):
        self.id = unique_id
        self.pos = starting_position
        self.old_pos_list = []
        self.trip_dest_node_id = trip_destination_node_id
        self.trip_dest_node_pos = trip_destination_position

        self.route_waypoints = trip_waypoints
        self.route_node_counter = 0
        self.num_route_nodes = len(self.route_waypoints)

        self.link_id = 0
        self.link_entry_ts = 0
        self.link_exit_ts = first_link_exit_time
        self.link_length = 0 #? maybe should be an input...
     
        self.link_entry_node_id = start_node_id
        self.link_exit_node_id = trip_waypoints[1]
        self.link_exit_node_pos =  next_node_enroute_pos
        self.dist_to_next_node = 1000 
     
        self.occupied = True
        self.t = 0 #personal timestep counter?


        self.passenger_trip_id = 0
        self.trip_start_ts = 0
        self.other_trip_loc = 0
        self.other_trip_ts = 0


class PassengerTrip():

    def __init__(self,unique_id, starting_position, start_node_id, trip_destination_position, trip_destination_node_id, trip_waypoints):

        self.id = unique_id

        self.start_node_id = start_node_id
        self.start_pos = starting_position
        
        self.dest_node_id = trip_destination_node_id        
        self.dest_pos = trip_destination_position

        self.route_waypoints = trip_waypoints

        self.pick_up_ts = 0
        self.drop_off_ts = 0
        self.trip_duration = 0

        self.served_by = None

# Some Useful Functions:

def NextLinkTaxiPos(taxi, MAX_V, road_network, link_dict):
    """ func to move taxi to the next link along it's planned route... AND updates link_dict's traffic count"""

    #update link traffic count dict.
    #remove vehicle from old link?
    if link_dict[(taxi.link_entry_node_id,taxi.link_exit_node_id)]['traffic_count'] >0:
        link_dict[(taxi.link_entry_node_id,taxi.link_exit_node_id)]['traffic_count']-=1
    else:
        print('we have some serious issues, link: (%i,%i) appears to have negative traffic volume...' % (taxi.link_entry_node_id, taxi.link_exit_node_id))

    #in the near future... if this link is too congested... maybe re-route?

    #now shift taxi to new link...
    taxi.route_node_counter +=1
    taxi.link_entry_node_id = taxi.link_exit_node_id
    taxi.link_exit_node_id = taxi.route_waypoints[taxi.route_node_counter]
    
    taxi.link_length = road_network[taxi.link_entry_node_id][taxi.link_exit_node_id][0]['length']
    taxi.dist_to_next_node = taxi.link_length

    taxi.link_entry_ts = taxi.t #note we should have already +=1'd this phucker at start of 'big loop'
    taxi.link_transit_time = int(tax.link_length/MAX_V)
    taxi.link_exit_ts = taxi.link_entry_ts + taxi.link_transit_time
    
    taxi.pos = [road_network.nodes[taxi.link_entry_node_id]['x'], road_network.nodes[taxi.link_entry_node_id]['y']]


    #update link traffic count dict, now adding taxi to new link....
    link_dict[(taxi.link_entry_node_id,taxi.link_exit_node_id)]['traffic_count']+=1

    return taxi, link_dict



def StraightLineInterp(x1,y1,t1,x2,y2,t2,T):
    dt = t2-t1
    dT = T-t1
    xT = dT*(x2-x1)/dt + x1
    yT = dT*(y2-y1)/dt + y1

    return round(xT,6), round(yT,6)

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

def RandomNewRoute2(road_network, node_id_array, current_node_id):

    internal_func_node_id_list = list(node_id_array)
    internal_func_node_id_list.remove(current_node_id)

    new_target_node_id = np.random.choice(internal_func_node_id_list)

    taxi_has_path = has_path(road_network, current_node_id, new_target_node_id)

    if taxi_has_path is False:
        MAX_ITER = 10
        ITER =0

        while ITER<MAX_ITER and taxi_has_path is False:
            new_target_node_id = np.random.choice(internal_func_node_id_list)
            taxi_has_path = has_path(road_network, current_node_id, new_target_node_id)
            ITER+=1

    if taxi_has_path is False and ITER==MAX_ITER:
    # randomly select new starting node... and repeat process...
        print("taxi is stuck at node %i, will be shot and replaced by a new taxi" % (current_node_id))


    return new_target_node_id, taxi_has_path

























#inputs for simulation...
#C225!

data_file_path =  '/home/toshiba/ABMTANET/gpickle_road_network_data/' 


graph_data_filename = 'Highest_Protocol_Roma_Centrale_Road_Network.gpickle' 
results_filename = 'roma_centrale_test_sim_4hrs.pickle'
CITY_NAME = 'Roma'
SIM_RUN_DATE = '3000taxis_0.5hr_v2vdata_15feb'
passenger_trip_data_filename = 'roma_passenger_trip_data_dict_12feb.pickle'
sensor_data_filename = 'roma_sensor_pos_dict_12feb.pickle' #'roma_sensor_pos_dict_21Jan.pickle'




"""
#San Fran-phuckin-cisco
CITY_NAME = 'sf'
SIM_RUN_DATE = '1000taxis_0.5hrs_v2vdata_14feb'
passenger_trip_data_filename = 'sf_passenger_trip_data_dict_28Jan.pickle'
sensor_data_filename = 'sf_sensor_pos_dict_28Jan.pickle'

graph_data_filename = 'Highest_Protocol_SF_Central_Road_Network.gpickle'
"""


"""
CITY_NAME = 'birmingham'
SIM_RUN_DATE = '2000taxis_0.5hrs_v2vdata_14feb'

passenger_trip_data_filename = 'birmingham_passenger_trip_data_dict_28Jan.pickle'
sensor_data_filename = 'birmingham_sensor_pos_dict_28Jan.pickle'
graph_data_filename = 'Highest_Protocol_Birmingham_central_Road_Network.gpickle'

"""



################# GENERAL SIMULATION PARAMS and Input DATA....
output_results_data_file_path =  '/home/toshiba/ABMTANET/simulation_results/'




road_network = read_gpickle(data_file_path + graph_data_filename)

node_id_list = list(road_network.nodes()) 
edge_list = list(road_network.edges())

link_dict = dict()
for g in range(0,len(edge_list)):
    link_dict[(edge_list[g][0],  edge_list[g][1])] = {'len':road_network[edge_list[g][0]][edge_list[g][1]][0]['length'], 'traffic_count':0}

node_longitude_dict = dict(road_network.nodes(data='x'))
node_latitude_dict = dict(road_network.nodes(data='y'))
    # map boundaries....
MAX_LONG = max(node_longitude_dict.values())
MIN_LONG  = min(node_longitude_dict.values())
MAX_LAT = max(node_latitude_dict.values())
MIN_LAT =  min(node_latitude_dict.values())

model_space_width_m = HaversineDistPC2([MAX_LONG, MAX_LAT],[MIN_LONG, MAX_LAT])
model_space_height_m = HaversineDistPC2([MAX_LONG,MIN_LAT],[MAX_LONG, MAX_LAT])


# Some key model parameters:
LEN_SIM = int(60*60*0.5) #simulation length, in real seconds, 1s=1sim_Step

NUM_TAXIS = 3000
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


#from pre-generated passenger trip file/pickle...
passenger_trip_list = []
taxi_id_counter = 0
taxi_agent_list = []

taxi_longitude_array = np.zeros(NUM_TAXIS)
taxi_latitude_array = np.zeros(NUM_TAXIS)

passenger_trip_id_counter = 0
start_passenger_trip_longitude_array = np.zeros(NUM_TRIPS)
start_passenger_trip_latitude_array =np.zeros(NUM_TRIPS)

with open((data_file_path+passenger_trip_data_filename),'rb') as handle01:
        passenger_trip_dict = pickle.load(handle01)
handle01.close()



pregenerated_trip_list = list(passenger_trip_dict.keys())

# Assigning trips to taxis and passenger agent classes...
for key, values in passenger_trip_dict.items():

        if key<NUM_TRIPS:
            #passenger_trip_id_counter +=1
            new_passenger_trip = PassengerTrip(key, values['start_pos'], values['start_node'], values['dest_pos'], values['dest_node'], values['route'])

            passenger_trip_list.append(new_passenger_trip)
            start_passenger_trip_longitude_array[key] = values['start_pos'][0]
            start_passenger_trip_latitude_array[key] = values['start_pos'][1]

        if key>=NUM_TAXIS and taxi_id_counter<NUM_TAXIS:

            
            first_link_exit_time = road_network[values['route'][0]][values['route'][1]][0]['length']/MAX_V

            link_exit_node_pos = [node_longitude_dict[values['route'][1]],node_latitude_dict[values['route'][1]]]
         
            new_taxi_agent = TaxiAgent(taxi_id_counter, values['start_pos'], values['start_node'], values['dest_pos'], values['dest_node'], values['route'], link_exit_node_pos, first_link_exit_time)

            taxi_longitude_array[taxi_id_counter] = new_taxi_agent.pos[0]
            taxi_latitude_array[taxi_id_counter] = new_taxi_agent.pos[1]

            taxi_agent_list.append(new_taxi_agent)
            taxi_id_counter +=1


    #each taxi needs to be recorded in our link_dict traffic count...
            if link_dict[(new_taxi_agent.link_entry_node_id, new_taxi_agent.link_exit_node_id)]['traffic_count'] == 0:
               link_dict[(new_taxi_agent.link_entry_node_id, new_taxi_agent.link_exit_node_id)]['traffic_count'] +=1



taxi_message_set_dict = dict()
sensor_message_set_dict = dict()

for sensor_id in range(NUM_SENSORS):
    sensor_message_set_dict[sensor_id] = {sensor_id}
for taxi_id in range(NUM_TAXIS):
    taxi_message_set_dict[taxi_id]= set()



# Simulation Initiaion...

node_id_array = np.array(node_id_list)

passenger_trip_results_dict = dict()
taxi_route_break_dict = dict()
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
    taxis_trips_dist_mat = HaversineDistMatrix(start_passenger_trip_latitude_array, start_passenger_trip_longitude_array, taxi_latitude_array, taxi_longitude_array)

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


    # okay, nearly there with this passenger-trip bullshit... still need to find taxis-passenger_trips within range, and then assign the trips to the taxis that are empty... 
    passenger_taxis_within_range_index = np.where((taxis_trips_dist_mat>0) & (taxis_trips_dist_mat<=V2I_MAXRANGE))
    #taxis along the horizontal[1], trips down the vertical[0]

    for j in range(len(passenger_taxis_within_range_index[0])):

        taxi_agent = taxi_agent_list[passenger_taxis_within_range_index[1][j]]
        trip_agent = passenger_trip_list[passenger_taxis_within_range_index[0][j]]

        if taxi_agent.occupied is False and trip_agent.served_by is None:

            #TAXI agent first...
            taxi_agent.occupied = True
            taxi_agent.trip_start_ts = time_step

            taxi_agent.trip_dest_node_id = trip_agent.dest_node_id
            taxi_agent.trip_dest_node_pos = trip_agent.dest_pos 

            taxi_agent.route_node_counter = 0
            taxi_agent.route_waypoints = trip_agent.route_waypoints
            taxi_agent.num_route_nodes = len(taxi.route_waypoints)

            taxi_agent.passenger_trip_id = trip_agent.id

            #TRIP agent...
            trip_agent.pick_up_ts = time_step
            trip_agent.served_by = taxi_agent.id

            print('taxi: %i is now serving trip: %i going from: %i to %i in %i steps' % (taxi_agent.id, trip_agent.id, trip_agent.start_node_id, taxi_agent.trip_dest_node_id, taxi_agent.num_route_nodes))


    # section regards sharing spare trip information, not necessary now... 
        """
        if taxi_agent.occupied is True and trip_agent.served_by is not None: #add trip info to taxi memory, maybe store only latest trip it passed but could not serve as it was already serving another passenger trip...
            taxi_agent.other_trip_loc = [start_passenger_trip_latitude_array[trip_agent.id], start_passenger_trip_longitude_array[trip_agent.id]]
            taxi_agent.other_trip_ts = time_step
        """

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

    for taxi in taxi_agent_list:

        #taxi.t +=1
        taxi.t = time_step

        taxi.dist_to_next_node = HaversineDistPC2(taxi.pos, taxi.link_exit_node_pos)


        # Check if taxi is near/about to arrive at target/destination?
        if taxi.link_exit_node_id != taxi.trip_dest_node_id:

            # Check if near to next node en route... 
            if taxi.dist_to_next_node>5 and taxi.link_exit_ts-taxi.t>1:

                new_taxi_pos = StraightLineInterp(taxi.pos[0], taxi.pos[1], taxi.link_entry_ts, taxi.link_exit_node_pos[0], taxi.link_exit_node_pos[1], taxi.link_exit_ts, taxi.t)

                taxi.pos = new_taxi_pos


            if taxi.dist_to_next_node<=5 or taxi.link_exit_ts-taxi.t<1:

                taxi.route_node_counter +=1

                taxi.link_entry_node_id = taxi.route_waypoints[taxi.route_node_counter]
                taxi.link_exit_node_id = taxi.route_waypoints[taxi.route_node_counter + 1]

                taxi.link_length = road_network[taxi.link_entry_node_id][taxi.link_exit_node_id][0]['length']

#                taxi.link_length = link_dict[(taxi.link_entry_node_id, taxi.link_exit_node_id)]['len']

                taxi.dist_to_next_node = taxi.link_length

                taxi.link_exit_node_pos = [road_network.node[taxi.link_exit_node_id]['x'], road_network.node[taxi.link_exit_node_id]['y']]

                taxi.link_entry_ts = taxi.t

                taxi.link_transit_time = int(taxi.link_length/MAX_V)
                taxi.link_exit_ts = taxi.link_entry_ts + taxi.link_transit_time

                taxi.pos = [road_network.nodes[taxi.link_entry_node_id]['x'], road_network.nodes[taxi.link_entry_node_id]['y']]




        if taxi.link_exit_node_id == taxi.trip_dest_node_id and taxi.dist_to_next_node>5 and taxi.link_exit_ts-taxi.t>1:           

            taxi.pos = StraightLineInterp(taxi.pos[0], taxi.pos[1], taxi.link_entry_ts, taxi.link_exit_node_pos[0], taxi.link_exit_node_pos[1], taxi.link_exit_ts, taxi.t)


        if taxi.link_exit_node_id == taxi.trip_dest_node_id and (taxi.dist_to_next_node<=5 or taxi.link_exit_ts-taxi.t<1):

            # Check if taxi was carrying  a passenger?
            if taxi.occupied is True:
                trip_agent = passenger_trip_list[taxi.passenger_trip_id]
                trip_agent.drop_off_ts = time_step
                passenger_trip_time = time_step - trip_agent.pick_up_ts
                trip_agent.trip_duration = passenger_trip_time
                print('taxi: %i has served trip: %i in %i [s]' % (taxi.id, taxi.passenger_trip_id, passenger_trip_time))

                # Record passenger trip data:
                passenger_trip_results_dict[trip_agent.id] = {'dur':passenger_trip_time,'taxi':taxi.id,'start_node':trip_agent.start_node_id,'dest_node':trip_agent.dest_node_id,'pickup_ts':trip_agent.pick_up_ts}
                
                taxi.occupied = False

            if taxi.occupied is False:
                print('taxi: %i has completed empty rando trip' % (taxi.id))
                
                # Getting into the Jesus of the matter... 
                # Do a quick search to see if you can assign a new trip within system, otherwise... select one from pre-approved list... taxi may disappear and re-appear allow...

                new_target_node_id, taxi_has_path_check = RandomNewRoute2(road_network, node_id_array, taxi.link_exit_node_id)

                if taxi_has_path_check is True:
                    taxi.link_entry_node_id = taxi.link_exit_node_id
                    print('Taxi %i has been assinged NEW trip: %i to %i' % (taxi.id, taxi.link_exit_node_id, new_target_node_id))

                    taxi.trip_dest_node_id = new_target_node_id
                    taxi.trip_dest_node_pos = [road_network.node[taxi.trip_dest_node_id]['x'], road_network.node[taxi.trip_dest_node_id]['y']]

                    taxi.route_node_counter = 0
                    taxi.route_waypoints = shortest_path(road_network,taxi.link_entry_node_id, taxi.trip_dest_node_id,weight='length')
                    taxi.num_route_nodes = len(taxi.route_waypoints)

                    taxi.link_entry_ts = time_step
                    taxi.link_exit_node_id = taxi.route_waypoints[1]
                    taxi.link_exit_node_pos = [road_network.node[taxi.route_waypoints[1]]['x'], road_network.node[taxi.route_waypoints[1]]['y']]
                    taxi.link_length = road_network[taxi.link_entry_node_id][taxi.link_exit_node_id][0]['length']

                    taxi.link_transit_time = int(taxi.link_length/MAX_V)
                    taxi.link_exit_ts = taxi.link_entry_ts + taxi.link_transit_time

                    taxi.dist_to_next_node = taxi.link_length
                    taxi.occupied = False

                if taxi_has_path_check is False:
                    # extreme phuck up case, just pick random pre-approved/generated trip from list...
                    new_rando_trip_id = np.random.choice(pregenerated_trip_list)
                    taxi.passenger_trip_id = new_rando_trip_id 
                    taxi.pos = passenger_trip_dict[new_rando_trip_id]['start_pos']
                    taxi.link_entry_node_id = passenger_trip_dict[new_rando_trip_id]['start_node']
                    taxi.trip_dest_node_pos = passenger_trip_dict[new_rando_trip_id]['dest_pos']
                    taxi.trip_dest_node_id = passenger_trip_dict[new_rando_trip_id]['dest_node']
                    taxi.route_waypoints = passenger_trip_dict[new_rando_trip_id]['route']

                    taxi.num_route_nodes = len(taxi.route_waypoints)
                    taxi.route_node_counter = 0
                    taxi.link_entry_ts = time_step
                    taxi.link_exit_node_id = taxi.route_waypoints[1]
                    taxi.link_exit_node_pos = [road_network.node[taxi.route_waypoints[1]]['x'], road_network.node[taxi.route_waypoints[1]]['y']]
                    taxi.link_length = road_network[taxi.link_entry_node_id][taxi.link_exit_node_id][0]['length']
                    taxi.dist_to_next_node = taxi.link_length
                    taxi.link_transit_time = int(taxi.link_length/MAX_V)
                    taxi.link_exit_ts = taxi.link_entry_ts + taxi.link_transit_time
                    taxi.occupied = False

                    print('taxi: %i has been assigned a pre-generated route (%i) from: %i to %i in %i waypoints... ' % (taxi.id, new_rando_trip_id, taxi.link_entry_node_id, taxi.trip_dest_node_id, taxi.num_route_nodes))




        if taxi.link_exit_ts-taxi.link_entry_ts<2 and taxi.link_length<10:
            taxi.link_exit_ts = taxi.link_exit_ts +1
            #print('taxi: %i needed some corrective timings...' % taxi.id)


        taxi_longitude_array[taxi.id] = taxi.pos[0]
        taxi_latitude_array[taxi.id] = taxi.pos[1]


    CIA_memory_dict[time_step] = {'lon':taxi_longitude_array, 'lat':taxi_latitude_array}







end_code_time = timer()

print('%i iterations in: ' % LEN_SIM)
print(end_code_time-start_code_time)




with open((output_results_data_file_path +('%s_cluster_taxi_messages_results_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle:
    pickle.dump(taxi_message_set_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open((output_results_data_file_path +('%s_sensor_messages_results_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle2:
    pickle.dump(sensor_message_set_dict, handle2, protocol = pickle.HIGHEST_PROTOCOL)

sensor_locs_dict = {'lon':sensor_longitude_array, 'lat':sensor_latitude_array}
with open((output_results_data_file_path +('%s_sensor_locations_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle3:
    pickle.dump(sensor_locs_dict, handle3, protocol = pickle.HIGHEST_PROTOCOL)

#with open((output_results_data_file_path +('%s_link_traffic_count_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle3:
#    pickle.dump(link_dict, handle3, protocol = pickle.HIGHEST_PROTOCOL)


with open((output_results_data_file_path +('%s_passenger_trip_results_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle4:
    pickle.dump(passenger_trip_results_dict, handle4, protocol = pickle.HIGHEST_PROTOCOL)
handle4.close()

#save general model details to make plotting easier?
general_model_params_dict = {'city':CITY_NAME,'model_width':model_space_width_m, 'model_height':model_space_height_m, 'sim_len':LEN_SIM, 'num_taxis':NUM_TAXIS, 'num_sensors':NUM_SENSORS, 'num_trips':NUM_TRIPS, 'max_v':MAX_V, 'min_lat': MIN_LAT, 'max_lat':MAX_LAT, 'min_lon':MIN_LONG,'max_lon':MAX_LONG}

with open((output_results_data_file_path +('%s_general_model_params_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle5:
    pickle.dump(general_model_params_dict, handle5, protocol = pickle.HIGHEST_PROTOCOL)
handle5.close()


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
passenger_trip_start_locs_dict = {'latitude': start_passenger_trip_latitude_array, 'longitude':start_passenger_trip_longitude_array}

with open((output_results_data_file_path+('%s_passenger_trip_start_locs_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle7:
    pickle.dump(passenger_trip_start_locs_dict, handle7, protocol = pickle.HIGHEST_PROTOCOL)
handle7.close()

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






























       
































