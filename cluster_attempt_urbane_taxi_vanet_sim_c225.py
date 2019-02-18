"""

focusing now on the cluster side of things...
code runs well with speed up, 'forked' from jan9 re_re_routing_urban_taxi_vanet_sim.py

speed-ups:
- [x] taxis do not need to be re-routing/searching for a fastest/shortest path... only do so if new information has "come to light"
{speed-up factor circa 50}
(too much effort...
- pre-compute taxi positions at link entry?
- python multiprocessing tool...
)

model functionality:
- [~x~] link transit time function
    nearly there, need to have an actual function, and probably add some pseudo-traffic to make it more interesting?
    maybe add a congested factor? link_length/(num_vehicles*5metres), ensure it can never get greater than 1

- taxi link transit time sharing and re-routing/searching for faster/optimal routes
- V2V success probability model...

Results/data collection
- packet transit/end-to-end delay needs recording...
- trip waiting, trip length times...
- Record where V2V exchanges most likely?
- How many V2V hops per successful message delivery?


Cluster issues...
- still not installed libgeos_c library... (could be sorted by re-installing shapely 3?)
- although, appears Pyinstaller is using python2.7... which is highly awkward.

./cluster_attempt_urbane_taxi_vanet_sim -s ~/MiniTaxiFleets/bigloop/ -j Cologne_Centraal_Road_Network.gpickle



maybe consider storing pre-computed starting trips and trip demand (i.e. those that have been checked for having a 'path'....
this could be saved and loaded everytime, should speed up debugging?




why are the packet delivery ratios sooooo low? like less than 1%??????
tried longer 8hr simulations (500 taxis, trips and sensors)... not much...
now trying 2000 taxis, trips and 500 sensors, same 64km2 map of Roma...


"""


#timer? IHKH
from timeit import default_timer as timer

# Import key libraries...
import numpy as np
import shapely #??? why is this needed so badly?
from networkx import has_path, shortest_path, read_gpickle
import pickle

#Results Plotting
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import cm
#plt.ion()

# Some useful classes...
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


    def StraightLineInterp(self,x1,y1,t1,x2,y2,t2,T):
        dt = t2-t1
        dT = T-t1
        xT = dT*(x2-x1)/dt + x1
        yT = dT*(y2-y1)/dt + y1

        return round(xT,6), round(yT,6)

    def HaversineDistPC2(self,pos1,pos2):
        #where pos1 & pos2 are tuples: (longitude,latitude)
        lon1, lat1, lon2, lat2 = map(np.radians, [pos1[0],pos1[1],pos2[0], pos2[1]])
        dlon = lon2-lon1
        dlat = lat2-lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        Hdistance = 6371e3*c  #working in metres!
        return int(Hdistance)







# Some useful functions


#

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
        print("taxi %i is stuck at node %i, will be shot and replaced by a new taxi")


    return new_target_node_id, taxi_has_path


# Road Network (import from osmnx/gpickle)
#C207 debugging...
"""
# small Sacremento City map
data_file_path = '/home/user/MiniTaxiFleets/osmnxtoexe/' #results.graph_filepath
graph_data_filename = 'test.gpickle' #results.graph_filename
results_filename = 'second_bigloop_graph_cluster_trial.pcikle'
"""

"""
#still C207
#a bigger map... Cologne Centraaaal...
#data_file_path = '/home/user/MiniTaxiFleets/bigloop/' #results.graph_filepath
#graph_data_filename = 'Highest_Protocol_Cologne_Centraal_Road_Network.gpickle' #results.graph_filename
#results_filename = 'HP_Cologne_Centraal.pickle'

#even BIGGER map, 64km^2 biatch.
data_file_path = '/home/user/ABMTANET/gpickle_road_network_data/' #results.graph_filepath
#graph_data_filename = 'Highest_Protocol_SF_Central_Road_Network.gpickle' #results.graph_filename
#results_filename = 'SF_central_test_sim.pickle'
graph_data_filename = 'Highest_Protocol_Roma_Centrale_Road_Network.gpickle' #results.graph_filename
#results_filename = 'roma_centrale_test_sim_4hrs.pickle'

passenger_trip_data_filename = 'roma_passenger_trip_data_dict_18Jan.pickle'
sensor_data_filename = 'roma_sensor_pos_dict_18Jan.pickle'

CITY_NAME = 'Roma'
SIM_RUN_DATE = '021Jan'

output_results_data_file_path =  '/home/user/ABMTANET/simulation_results/'
"""


#C255!
#running long-ass simulations... 4+ hours...
data_file_path =  '/home/toshiba/ABMTANET/gpickle_road_network_data/' #'/home/toshiba/MiniTaxiFleets/bigloop/'
graph_data_filename = 'Highest_Protocol_SF_Central_Road_Network.gpickle' #results.graph_filename
#graph_data_filename = 'Highest_Protocol_Roma_Centrale_Road_Network.gpickle' #results.graph_filename
#results_filename = 'roma_centrale_test_sim_4hrs.pickle'

#CITY_NAME = 'Roma'
#SIM_RUN_DATE = '2Ktaxis_4hrs_v2vdata_25jan'

CITY_NAME = 'sf'
SIM_RUN_DATE = '2Ktaxis_4hrs_v2vdata_28jan'

passenger_trip_data_filename = 'sf_passenger_trip_data_dict_28Jan.pickle'
sensor_data_filename = 'sf_sensor_pos_dict_28Jan.pickle'
output_results_data_file_path =  '/home/toshiba/ABMTANET/simulation_results/'


"""
# when compiling... input arguments...
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', action='store', type=str, dest='graph_filepath', help='where is this phucking pickled graph data?')
parser.add_argument('-j', action='store', type=str, dest='graph_filename', help='what is graph filename?')
results = parser.parse_args()

data_file_path = results.graph_filepath
graph_data_filename = results.graph_filename
"""




road_network = read_gpickle(data_file_path + graph_data_filename)

node_id_list = list(road_network.nodes()) 
edge_list = list(road_network.edges())

link_dict = dict()
for j in range(0,len(edge_list)):
    link_dict[(edge_list[j][0],  edge_list[j][1])] = {'len':road_network[edge_list[j][0]][edge_list[j][1]][0]['length'], 'traffic_count':0}

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
LEN_SIM = 60*60*4 #simulation length, in real seconds, 1s=1sim_Step

NUM_TAXIS = 2000 #500 #15*64 # ?15 taxis per km^2
NUM_SENSORS = 500
NUM_TRIPS = 2000 #500

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
with open((data_file_path + sensor_data_filename),'rb') as handle:
    sensor_pos_dict = pickle.load(handle)

sensor_longitude_array = sensor_pos_dict['longitude']
sensor_latitude_array = sensor_pos_dict['latitude']





#TRIPS. could be randomly placed, however, problems occur with routing... hence maybe also locate them at intersections/nodes... for now before some serious 'snapping-function' is made...


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



#from pre-generated passenger trip file/pickle...

passenger_trip_list = []
taxi_id_counter = 0
taxi_agent_list = []

taxi_longitude_array = np.zeros(NUM_TAXIS)
taxi_latitude_array = np.zeros(NUM_TAXIS)

passenger_trip_id_counter = 0
start_passenger_trip_longitude_array = np.zeros(NUM_TRIPS)
start_passenger_trip_latitude_array =np.zeros(NUM_TRIPS)

with open((data_file_path+passenger_trip_data_filename),'rb') as handle:
        passenger_trip_dict = pickle.load(handle)

for key, values in passenger_trip_dict.items():

        if key<NUM_TAXIS:
            #passenger_trip_id_counter +=1
            new_passenger_trip = PassengerTrip(key, values['start_pos'], values['start_node'], values['dest_pos'], values['dest_node'], values['route'])

            passenger_trip_list.append(new_passenger_trip)
            start_passenger_trip_longitude_array[key] = values['start_pos'][0]
            start_passenger_trip_latitude_array[key] = values['start_pos'][1]

        if key>NUM_TAXIS:

            
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


"""
#this code generates trips witin the script...

passenger_trip_list = []
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

        new_passenger_trip = PassengerTrip(passenger_trip_id, passenger_trip_start_pos, trip_start_node_id, passenger_trip_destination_pos, trip_target_node_id, passenger_trip_waypoints)

        passenger_trip_list.append(new_passenger_trip)

"""


# taxis will initially start at nodes, this is just for debugging...
#taxi_longitude_array = (np.random.rand(NUM_TAXIS,)*(MAX_LONG-MIN_LONG))+MIN_LONG
#taxi_latitude_array = (np.random.rand(NUM_TAXIS,)*(MAX_LAT-MIN_LAT))+MIN_LAT

#link_transit_times


# setting up initial taxi positions, note taxis start at intersections!



"""
# original code for generating initial taxi trips... regardless of passenger status

taxi_message_set_dict = dict()
sensor_message_set_dict = dict()

for sensor_id in range(NUM_SENSORS):
    sensor_message_set_dict[sensor_id] = {sensor_id}
for taxi_id in range(NUM_TAXIS):
    taxi_message_set_dict[taxi_id]= set()


taxi_start_node_ids_list = []
taxi_dest_node_ids_list = []

for taxi_agent in range(NUM_TAXIS):
    
    taxi_start_node_id = np.random.choice(node_id_list)
    taxi_target_node_id = np.random.choice(node_id_list)
    while taxi_start_node_id == taxi_target_node_id:
        taxi_target_node_id = np.random.choice(node_id_list)

    taxi_has_path = has_path(road_network, taxi_start_node_id, taxi_target_node_id)
    while taxi_has_path is False:
        print('we have a dud path, taxi id: %i, nodes: %i, %i' % (taxi_agent, taxi_start_node_id, taxi_target_node_id))
        taxi_target_node_id = np.random.choice(node_id_list)
        taxi_start_node_id = np.random.choice(node_id_list)
        taxi_has_path = has_path(road_network, taxi_start_node_id, taxi_target_node_id)

# record randomly assigned initial 'trips'
    taxi_start_node_ids_list.append(taxi_start_node_id)
    taxi_dest_node_ids_list.append(taxi_target_node_id)

# set up for future message/info exchange between taxis...
#    taxi_message_set_dict[taxi_agent] = set() #empty for now
#    taxi_transit_link_times_dict[taxi_agent] = dict() #{(Anode_id,Bnode_id):'trans_t':?,
# Setting up taxi agents...
taxi_agent_list = []
taxi_longitude_array = np.zeros(NUM_TAXIS,)
taxi_latitude_array = np.zeros(NUM_TAXIS,)

for i in range(NUM_TAXIS):

    trip_start_node_id = taxi_start_node_ids_list[i]
    trip_dest_node_id = taxi_dest_node_ids_list[i]

    trip_waypoints = shortest_path(road_network,trip_start_node_id, trip_dest_node_id,weight='length')

    starting_pos = [road_network.node[trip_start_node_id]['x'],road_network.node[trip_start_node_id]['y']]
    trip_destination_pos = [road_network.node[trip_dest_node_id]['x'],road_network.node[trip_dest_node_id]['y']]

    taxi_longitude_array[i] = starting_pos[0]
        first_link_exit_time = road_network[trip_waypoints[0]][trip_waypoints[1]][0]['length']/MAX_V
    new_taxi_agent = TaxiAgent(i, starting_pos, trip_start_node_id, trip_destination_pos, trip_dest_node_id, trip_waypoints, next_node_enroute_pos, first_link_exit_time)

    taxi_agent_list.append(new_taxi_agent)taxi_latitude_array[i] = starting_pos[1]

    next_node_enroute_id = trip_waypoints[1]
    next_node_enroute_pos =  [road_network.node[next_node_enroute_id]['x'],road_network.node[next_node_enroute_id]['y']]

    first_link_exit_time = road_network[trip_waypoints[0]][trip_waypoints[1]][0]['length']/MAX_V
    new_taxi_agent = TaxiAgent(i, starting_pos, trip_start_node_id, trip_destination_pos, trip_dest_node_id, trip_waypoints, next_node_enroute_pos, first_link_exit_time)

    taxi_agent_list.append(new_taxi_agent)



    #each taxi needs to be recorded in our link_dict traffic count...


    if link_dict[(new_taxi_agent.link_entry_node_id, new_taxi_agent.link_exit_node_id)]['traffic_count'] == 0:
       link_dict[(new_taxi_agent.link_entry_node_id, new_taxi_agent.link_exit_node_id)]['traffic_count'] +=1

"""





#due to disapearing nodes... apparently even within funcs, list.remove() applies to everything, not just func. internals..... fawkward.
debug_node_id_list_len = []

node_id_array = np.array(node_id_list)


start_code_time = timer()

passenger_trip_results_dict = dict()
taxi_route_break_dict = dict()



slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LONG,MAX_LONG,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)


v2v_sharing_loc_dict = dict()
v2v_edge_data_dict = dict()




for time_step in range(0,LEN_SIM):

    print('simulation time step = %i' % time_step)
    #debug_node_id_list_len.append(len(node_id_list))
    #print('node id list length= %i' % debug_node_id_list_len[-1])
    # Distance Matricies evaluations...
    taxis_sensors_dist_mat =  HaversineDistMatrix(sensor_latitude_array, sensor_longitude_array, taxi_latitude_array, taxi_longitude_array)
    taxis_taxis_dist_mat = SelfHaversineDistMatrix(taxi_latitude_array, taxi_longitude_array)
    taxis_trips_dist_mat = HaversineDistMatrix(start_passenger_trip_latitude_array, start_passenger_trip_longitude_array, taxi_latitude_array, taxi_longitude_array)

    #Comms message exchange
    #V2I - taxi-sensors
    taxis_sensors_within_range_index = np.where(taxis_sensors_dist_mat<=V2I_MAXRANGE)
    for j in range(0,taxis_sensors_within_range_index[0].size):

        sensor_id = taxis_sensors_within_range_index[0][j]
        taxi_id = taxis_sensors_within_range_index[1][j]

        #print('taxi %i exchanged with sensor %i at t=%i' % (taxi_id, sensor_id, time_step))
        current_taxi_message_set = taxi_message_set_dict[taxi_id]
        current_sensor_message_set = sensor_message_set_dict[sensor_id]
        combined_message_sets = current_taxi_message_set.union(current_sensor_message_set)

        taxi_message_set_dict[taxi_id] = combined_message_sets
        sensor_message_set_dict[sensor_id] = combined_message_sets

    #V2I - taxi-trip?
    # when exchanging this... taxiA-TaxiB, if taxiA/B is unoccupied, and trip sensed nearby (e.g. with time limit and distance range, 5 mins and 500m?) then need
    # to re-route/set a route waypoints for that taxi to go pick up passengers
    # consequently... if no passengers there (already served before taxi arrives...)
    # then need to set rando-waypoints

    #V2P - Taxi-Passenger Trip...
    #if taxi is unoccupied... and within range of a passenger trip... assign it to such trip. Then need to sort out new destination target and route within corresponding taxi_agent


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

            #this is where a 'new_trip_function would be really cool...
            taxi_agent.trip_dest_node_id = trip_agent.dest_node_id #??????????? need to get this from the passenger_trip class.
            taxi_agent.trip_dest_node_pos = trip_agent.dest_pos #[road_network.node[taxi.trip_dest_node_id]['x'],road_network.node[taxi.trip_dest_node_id]['y']]

            taxi_agent.route_node_counter = 0
            taxi_agent.route_waypoints = trip_agent.route_waypoints #shortest_path(road_network,taxi.link_entry_node_id, taxi.trip_dest_node_id,weight='length')
            taxi_agent.num_route_nodes = len(taxi.route_waypoints)

            taxi_agent.passenger_trip_id = trip_agent.id

            #TRIP agent...
            trip_agent.pick_up_ts = time_step
            trip_agent.served_by = taxi_agent.id
            #maybe 'pop' this trip off the dist_mat list... just to reduce computations?
            print('taxi: %i is now serving trip: %i going from: %i to %i in %i steps' % (taxi_agent.id, trip_agent.id, trip_agent.start_node_id, taxi_agent.trip_dest_node_id, taxi_agent.num_route_nodes))

        if taxi_agent.occupied is True and trip_agent.served_by is not None: #add trip info to taxi memory, maybe store only latest trip it passed but could not serve as it was already serving another passenger trip...
            taxi_agent.other_trip_loc = [start_passenger_trip_latitude_array[trip_agent.id], start_passenger_trip_longitude_array[trip_agent.id]]
            taxi_agent.other_trip_ts = time_step




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

    for i in shuffle_index:

        Ataxi_id = taxis_taxis_within_range_index[0][i]
        Btaxi_id = taxis_taxis_within_range_index[1][i]

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

    #v2v_sharing_loc_dict[time_step] = {'latitude':v2v_sharing_latitudes_array, 'longitude':v2v_sharing_longitudes_array,'len':v2v_sharing_counter}

    #quick 2d histogram for v2v sharing exchange location recording...
    H, xedges2, yedges2 = np.histogram2d(v2v_sharing_longitudes_array,v2v_sharing_latitudes_array, bins=(Xedges, Yedges))
    v2v_sharing_loc_dict[time_step] = H



    v2v_edge_data_dict[time_step] = v2v_edge_data_list

    """ ORIGINAL V2V EXCHANGE CODE:


        # now randomise taxi-pair order...
        shuffle_index = np.arange(taxis_taxis_within_range_index[0].size)
        np.random.shuffle(shuffle_index)

        for i in shuffle_index:

            Ataxi_id = taxis_taxis_within_range_index[0][i]
            Btaxi_id = taxis_taxis_within_range_index[1][i]

            Ataxi_message_set = taxi_message_set_dict[Ataxi_id]
            Btaxi_message_set = taxi_message_set_dict[Btaxi_id]

            if len(Ataxi_message_set)+len(Btaxi_message_set)>0:
                combined_taxis_message_sets = Ataxi_message_set.union(Btaxi_message_set)

                taxi_message_set_dict[Ataxi_id] = combined_taxis_message_sets
                taxi_message_set_dict[Btaxi_id] = combined_taxis_message_sets
                print('taxiA %i exchanged with taxiB %i at t=%i' % (Ataxi_id, Btaxi_id, time_step))
                print(combined_taxis_message_sets)

    """




    # Update taxis positions NOW.

    for taxi in taxi_agent_list:

        taxi.t +=1
        taxi.old_pos_list.append(tuple(taxi.pos))
        taxi.dist_to_next_node = taxi.HaversineDistPC2(taxi.pos, taxi.link_exit_node_pos) #maybe start using routing distance instead? just in case...

        if taxi.link_exit_node_id != taxi.trip_dest_node_id:
          
            if taxi.dist_to_next_node >5:
                taxi.pos = taxi.StraightLineInterp(taxi.pos[0],taxi.pos[1],taxi.link_entry_ts, taxi.link_exit_node_pos[0], taxi.link_exit_node_pos[1], taxi.link_exit_ts, taxi.t)

            else: #i.e. taxi is LESS than 5 metres from link ending/ road intersection...

# enter new NextLinkFunction... 
                #update link traffic count dict.
                #remove vehicle from old link?

                if (taxi.link_entry_node_id,taxi.link_exit_node_id) in link_dict:

                    if link_dict[(taxi.link_entry_node_id,taxi.link_exit_node_id)]['traffic_count'] >0:
                        link_dict[(taxi.link_entry_node_id,taxi.link_exit_node_id)]['traffic_count']-=1
                    else:
                        print('we have some serious issues, link: (%i,%i) appears to have negative traffic volume...' % (taxi.link_entry_node_id, taxi.link_exit_node_id))

                    #in the near future... if this link is too congested... maybe re-route?

                    #now shift taxi to new link...
                    taxi.route_node_counter +=1
                    taxi.link_entry_node_id = taxi.route_waypoints[taxi.route_node_counter] #taxi.link_exit_node_id
                    taxi.link_exit_node_id = taxi.route_waypoints[taxi.route_node_counter+1]
                    
                    taxi.link_length = road_network[taxi.link_entry_node_id][taxi.link_exit_node_id][0]['length']

                    taxi.link_length = link_dict[(taxi.link_entry_node_id,taxi.link_exit_node_id)]['len']

                    taxi.dist_to_next_node = taxi.link_length
                    taxi.link_exit_node_pos = [road_network.node[taxi.link_exit_node_id]['x'],road_network.node[taxi.link_exit_node_id]['y']]

                    taxi.link_entry_ts = taxi.t #note we should have already +=1'd this phucker at start of 'big loop'
                    taxi.link_transit_time = int(taxi.link_length/MAX_V)
                    taxi.link_exit_ts = taxi.link_entry_ts + taxi.link_transit_time
                    
                    taxi.pos = [road_network.nodes[taxi.link_entry_node_id]['x'], road_network.nodes[taxi.link_entry_node_id]['y']]


                    #update link traffic count dict, now adding taxi to new link....
                    link_dict[(taxi.link_entry_node_id,taxi.link_exit_node_id)]['traffic_count']+=1

                else:
                    print('taxiID: %i on a weird/dead link: (%i,%i)' % (taxi.id, taxi.link_entry_node_id,taxi.link_exit_node_id))


        if taxi.link_exit_node_id == taxi.trip_dest_node_id and taxi.dist_to_next_node<5: #i.e. if the taxis next node is it's destination, and within 5m? asign it new empty route...
            #trip length/time?
            
            if taxi.occupied is True:            
                trip_agent2 = passenger_trip_list[taxi.passenger_trip_id]
                trip_agent2.drop_off_ts = time_step
                passenger_trip_time = trip_agent2.drop_off_ts - trip_agent2.pick_up_ts
                trip_agent2.trip_duration = passenger_trip_time
                
                print('taxi: %i has served trip: %i in %i seconds' % (taxi.id, taxi.passenger_trip_id, passenger_trip_time))
                taxi.occupied = False #? remove other trip data...

                passenger_trip_results_dict[trip_agent2.id] = {'dur':passenger_trip_time,'taxi':taxi.id,'start_node':trip_agent2.start_node_id,'dest_node':trip_agent2.dest_node_id,'pickup_ts':trip_agent2.pick_up_ts}

            else:
                print('taxi: %i has finished random assigned empty search trip' % taxi.id)
    #enter new empty/search route function!
            new_target_node_id, taxi_has_path_check = RandomNewRoute2(road_network, node_id_array, taxi.link_exit_node_id)

            if taxi_has_path_check is False:
                if taxi.id in taxi_route_break_dict:
                    taxi_route_break_dict[taxi.id].append(taxi.t)
                else:
                    taxi_route_break_dict[taxi.id] = list(taxi.t)

                new_start_node_id = np.random.choice(node_id_array)
                new_target_node_id, taxi_has_path_check = RandomNewRoute2(road_network, node_id_array, new_start_node_id)
                
                if taxi_has_path_check is True:
                    taxi.link_entry_node_id = new_start_node_id
                    print('taxi %i NEW path, between nodes %i and %i' % (taxi.id, new_start_node_id, new_target_node_id))

            taxi.trip_dest_node_id = new_target_node_id
            taxi.trip_dest_node_pos = [road_network.node[taxi.trip_dest_node_id]['x'],road_network.node[taxi.trip_dest_node_id]['y']]

            taxi.route_node_counter = 0
            taxi.route_waypoints = shortest_path(road_network,taxi.link_entry_node_id, taxi.trip_dest_node_id,weight='length')
            taxi.num_route_nodes = len(taxi.route_waypoints)




    #Update Central Intelligence Matrix (CIM)
    taxi_longitude_array[taxi.id] = taxi.pos[0]
    taxi_latitude_array[taxi.id] = taxi.pos[1]


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

# save taxi data stored in class variables...

taxi_hist_pos_dict = dict()

for taxi in taxi_agent_list:

    taxi_hist_pos_dict[taxi.id] = taxi.old_pos_list
 
with open((output_results_data_file_path +('%s_taxi_agent_position_data_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle6:
    pickle.dump(taxi_hist_pos_dict, handle6, protocol = pickle.HIGHEST_PROTOCOL)
handle6.close()


#with open((output_results_data_file_path +('%s_taxi_route_break_dict_%s.pickle' % (CITY_NAME,SIM_RUN_DATE))), 'wb') as handle5:
#    pickle.dump(taxi_route_break_dict, handle5, protocol = pickle.HIGHEST_PROTOCOL)

### save trip start points...
passenger_trip_start_locs_dict = {'latitude': start_passenger_trip_latitude_array, 'longitude':start_passenger_trip_longitude_array}

with open((output_results_data_file_path+('%s_passenger_trip_start_locs_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE))), 'wb') as handle7:
    pickle.dump(passenger_trip_start_locs_dict, handle7, protocol = pickle.HIGHEST_PROTOCOL)
handle7.close()

with open((output_results_data_file_path +'%s_v2v_sharing_loc_heatmap_dict_%s.pickle' %(CITY_NAME, SIM_RUN_DATE)), 'wb') as handle8:
    pickle.dump( v2v_sharing_loc_dict, handle8, protocol=pickle.HIGHEST_PROTOCOL)
handle8.close()

# v2v edge data....
with open((output_results_data_file_path +'%s_v2v_edge_data_dict_%s.pickle' %(CITY_NAME, SIM_RUN_DATE)), 'wb') as handle9:
    pickle.dump(v2v_edge_data_dict, handle9, protocol=pickle.HIGHEST_PROTOCOL)
handle9.close()




"""
#for those memory disaster moments...

for i in range(10001,12000):
    v2v_edge_data_dict.pop(i)


H = np.zeros((159,159))
#jesus h fuck. living on the fucking edge here of memory.
for key, values in v2v_sharing_loc_dict.items():

    H+= values

with open((output_results_data_file_path +'%s_v2v_sharing_loc_heatmap_dict_%s.pickle' %(CITY_NAME, SIM_RUN_DATE)), 'wb') as handle10:
    pickle.dump( H, handle10, protocol=pickle.HIGHEST_PROTOCOL)
handle10.close()

"""


"""
######
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

####

"""
