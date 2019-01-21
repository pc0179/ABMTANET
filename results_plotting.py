
import pickle
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
plt.ion()





#general_model_params_dict = {'city':CITY_NAME,'model_width':model_space_width_m, 'model_height':model_space_height_m, 'sim_len':LEN_SIM, 'num_taxis':NUM_TAXIS, 'num_sensors':NUM_SENSORS, 'num_trips':NUM_TRIPS, 'max_v':MAX_V}
#c207
#sim_data_results_filepath = '/home/user/MiniTaxiFleets/bigloop/cluster_results/'


#C225
#sim_data_results_filepath = '/home/toshiba/MiniTaxiFleets/bigloop/simulation_results/'

#C207
sim_data_results_filepath = '/home/user/ABMTANET/simulation_results/'

#general model params
CITY_NAME = 'Roma' #'SF'
SIM_RUN_DATE = '021Jan'


general_model_params_filename = '%s_general_model_params_%s.pickle' % (CITY_NAME,SIM_RUN_DATE)
 
with open((sim_data_results_filepath+general_model_params_filename),'rb') as handle:
    general_model_params_dict = pickle.load(handle)

NUM_SENSORS = general_model_params_dict['num_sensors']
NUM_TAXIS = general_model_params_dict['num_taxis']
NUM_TRIPS = general_model_params_dict['num_trips']

model_space_height_m = general_model_params_dict['model_height'] #2334
model_space_width_m =  general_model_params_dict['model_width'] #3362
simulation_length = general_model_params_dict['sim_len']



#sensor_results_filename = 'sensor_messages_results.pickle'
sensor_results_filename = '%s_sensor_messages_results_%s.pickle' % (CITY_NAME, SIM_RUN_DATE)
with open(sim_data_results_filepath+sensor_results_filename , 'rb') as handle:
    sensor_results_dict = pickle.load(handle)

sensor_pdr = []
for sensor_id, messages in sensor_results_dict.items():
    sensor_pdr.append(len(messages)/NUM_SENSORS)



#taxi_results_filename = 'taxi_messages_results.pickle'
taxi_results_filename = '%s_cluster_taxi_messages_results_%s.pickle' % (CITY_NAME,SIM_RUN_DATE)

with open(sim_data_results_filepath+taxi_results_filename, 'rb') as handle2:
    taxi_results_dict = pickle.load(handle2)

taxi_prr = []
for taxi_agent, messages in taxi_results_dict.items():
    taxi_prr.append(len(messages)/NUM_SENSORS)


# Sensor locations...
sensor_locations_filename = '%s_sensor_locations_%s.pickle' % (CITY_NAME,SIM_RUN_DATE)
with open(sim_data_results_filepath+sensor_locations_filename, 'rb') as handle3:
    sensor_locations_dict = pickle.load(handle3)

fig, ax = plt.subplots()
ax.scatter(sensor_locations_dict['lon'],sensor_locations_dict['lat'])
ax.set_ylabel('latitude')
ax.set_xlabel('longitude')
ax.set_title('sensor locations')
plt.show()


# passenger trip results...
passenger_trips_filename = '%s_passenger_trip_results_%s.pickle' % (CITY_NAME,SIM_RUN_DATE)
with open(sim_data_results_filepath+passenger_trips_filename, 'rb') as handle4:
    passenger_trip_dict = pickle.load(handle4)

passenger_trip_ids = list(passenger_trip_dict.keys())
trip_durations = []

for key, values in passenger_trip_dict.items():

    trip_durations.append(values['dur'])
    

fig, ax = plt.subplots()
ax.bar(passenger_trip_ids, trip_durations)
ax.set_ylabel('Time/s')
ax.set_xlabel('Trip IDs')
ax.set_title('Trip Durations')
plt.show()

#Taxi Agent data....
taxi_agent_data_filename = '%s_taxi_agent_position_data_%s.pickle' % (CITY_NAME,SIM_RUN_DATE)
with open(sim_data_results_filepath+taxi_agent_data_filename, 'rb') as handle4:
    taxi_pos_dict = pickle.load(handle4)

#from matplotlib.pyplot import cm
"""
plotting_colours = iter(plt.cm.rainbow(np.linspace(0,1,NUM_TAXIS)))

fig, ax = plt.subplots()

for taxi_id, values in taxi_pos_dict.items():
    plot_colour = next(plotting_colours)
    lon, lat = list(zip(*values))
    ax.plot(lon,lat,c=plot_colour)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('%s' % CITY_NAME)
plt.show()
"""

# with fewer taxis...
"""
minor issues arise due to 'jumps' in taxi locations... maybe need to sort this out... just let taxis die/disapear or keep searching for node that has path, if not just retrace steps???
"""

#Passenger trip start locations
passenger_trip_start_locs_filname = '%s_passenger_trip_start_locs_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE)
with open(sim_data_results_filepath+passenger_trip_start_locs_filname, 'rb') as handle5:
    passenger_trip_start_locs_dict = pickle.load(handle5)


a = 0
N = 500
fig, ax = plt.subplots()
plotting_colours = iter(plt.cm.rainbow(np.linspace(0,1,N-a)))
for i in range(a,N):
    plot_colour = next(plotting_colours)
    lon, lat = list(zip(*taxi_pos_dict[i]))
    ax.plot(lon,lat,c=plot_colour)

ax.plot(sensor_locations_dict['lon'],sensor_locations_dict['lat'],'ok')
ax.plot(passenger_trip_start_locs_dict['longitude'], passenger_trip_start_locs_dict['latitude'],'*g')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('%s' % CITY_NAME)
plt.show()




plot_sensor_ids = np.arange(0,NUM_SENSORS)

plot_taxis_ids = np.arange(0,NUM_TAXIS)

mode_sensor_pdr = stats.mode(sensor_pdr)

fig, ax = plt.subplots()
ax.bar(plot_sensor_ids, sensor_pdr)
ax.yaxis.set_ticks(np.arange(0,1,0.1))
ax.set_ylim([0,1])
ax.set_ylabel('Packet Delivery Ratio')
ax.set_xlabel('Sensor IDs')
ax.set_title('sim_len= %i, %s=%f km^2, num.taxis=%i' % (simulation_length, CITY_NAME, ((model_space_height_m*model_space_width_m)/1e6), NUM_TAXIS))
ax.axhline(np.mean(sensor_pdr), color='green', label='Mean PDR')
ax.axhline(np.median(sensor_pdr), color='red', label='Median PDR')
ax.axhline(mode_sensor_pdr.mode[0], color='black', label='Modal PDR')
ax.legend(loc='upper right')
ax.margins(0.05)
plt.show()


plot_taxis_ids = np.arange(0,NUM_TAXIS)
mode_taxi_prr = stats.mode(taxi_prr)
fig, ax = plt.subplots()
ax.bar(plot_taxis_ids, taxi_prr)
ax.yaxis.set_ticks(np.arange(0,1,0.1))
ax.set_ylim([0,1])
ax.set_ylabel('Packet Received Ratio')
ax.set_xlabel('Taxi IDs')
ax.set_title('sim len = %i, %s=%f km^2, num.taxis=%i, num.sensors=%i' % (simulation_length, CITY_NAME, ((model_space_height_m*model_space_width_m)/1e6), NUM_TAXIS, NUM_SENSORS))
ax.axhline(np.mean(taxi_prr), color='green', label='Mean PDR')
ax.axhline(np.median(taxi_prr), color='red', label='Median PDR')
ax.axhline(mode_taxi_prr.mode[0], color='black', label='Modal PDR')
ax.legend(loc='upper right')
ax.margins(0.05)
plt.show()

# plot road graph just to check what the hell is happening...
"""
import osmnx
from networkx import read_gpickle


roadnetwork_data_file_path =  '/home/toshiba/MiniTaxiFleets/gpickle_road_network_data/'
roadnetwork_filename = 'Highest_Protocol_%s_Centrale_Road_Network.gpickle' % CITY_NAME
road_network = read_gpickle(roadnetwork_data_file_path + roadnetwork_filename)

osmnx.plot_graph(road_network)
"""


"""
# okay now for something 'fancy' connected sensor components...
import networkx as nx

sensor_graph = nx.Graph()

for sensor_id, messages in sensor_results_dict.items():

    sensor_graph.add_node(sensor_id)
    

for sensor_id, messages in sensor_results_dict.items():

    for connected_sensor in messages:
        sensor_graph.add_edge(sensor_id,connected_sensor, weight=np.random.randint(11,14400))

nx.draw_circular(sensor_graph, with_labels=True)
plt.show()

"""
