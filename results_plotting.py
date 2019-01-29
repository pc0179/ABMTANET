
import pickle
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
plt.ion()





#general_model_params_dict = {'city':CITY_NAME,'model_width':model_space_width_m, 'model_height':model_space_height_m, 'sim_len':LEN_SIM, 'num_taxis':NUM_TAXIS, 'num_sensors':NUM_SENSORS, 'num_trips':NUM_TRIPS, 'max_v':MAX_V}
#c207
#sim_data_results_filepath = '/home/user/MiniTaxiFleets/bigloop/cluster_results/'


#C225
#sim_data_results_filepath = '/home/toshiba/ABMTANET/simulation_results/'

#C207
#sim_data_results_filepath = '/home/user/ABMTANET/simulation_results/'
sim_data_results_filepath = '/home/user/Dropbox/tanet_sim_results/simulation_results/'

#general model params

#CITY_NAME = 'Roma'
#SIM_RUN_DATE = '2Ktaxis_2hrs_expolos_23jan'

CITY_NAME = 'Roma'
SIM_RUN_DATE = '2Ktaxis_30mins_expolos_23jan'

CITY_NAME = 'Roma'
SIM_RUN_DATE = '8hrs_expolos_21jan'

general_model_params_filename = '%s_general_model_params_%s.pickle' % (CITY_NAME,SIM_RUN_DATE)
 
with open((sim_data_results_filepath+general_model_params_filename),'rb') as handle:
    general_model_params_dict = pickle.load(handle)

NUM_SENSORS = general_model_params_dict['num_sensors']
NUM_TAXIS = general_model_params_dict['num_taxis']
NUM_TRIPS = general_model_params_dict['num_trips']

MIN_LAT = general_model_params_dict['min_lat']
MAX_LAT = general_model_params_dict['max_lat']
MIN_LON = general_model_params_dict['min_lon']
MAX_LON = general_model_params_dict['max_lon']

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

"""
fig, ax = plt.subplots()
ax.scatter(sensor_locations_dict['lon'],sensor_locations_dict['lat'])
ax.set_ylabel('latitude')
ax.set_xlabel('longitude')
ax.set_title('sensor locations')
plt.show()
"""

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
N = NUM_TAXIS #500

fig, ax = plt.subplots()
plotting_colours = iter(plt.cm.rainbow(np.linspace(0,1,N-a)))
for i in range(a,N):
    plot_colour = next(plotting_colours)
    lon, lat = list(zip(*taxi_pos_dict[i]))
    ax.plot(lon,lat,c=plot_colour)

#ax.plot(sensor_locations_dict['lon'],sensor_locations_dict['lat'],'ok')
#ax.plot(passenger_trip_start_locs_dict['longitude'], passenger_trip_start_locs_dict['latitude'],'*g')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('%s' % CITY_NAME)
plt.show()


#### V2V exchanges? and taxi trajectories...
v2v_exchanges_filname = '%s_v2v_sharing_loc_heatmap_dict_%s.pickle' % (CITY_NAME, SIM_RUN_DATE)

with open(sim_data_results_filepath+v2v_exchanges_filname, 'rb') as handle6:
    v2v_exchange_locs_dict = pickle.load(handle6)


a = 0
N = 500
#fig, ax = plt.subplots()
#plotting_colours = iter(plt.cm.rainbow(np.linspace(0,1,N-a)))
#for i in range(a,N):
#    plot_colour = next(plotting_colours)
#    lon, lat = list(zip(*taxi_pos_dict[i]))
#    ax.plot(lon,lat,c=plot_colour)


#plt.figure()
#plt.plot(lon2,lat2,'+k')
#plt.xlabel('longitude')
#plt.ylabel('latitude')
#plt.show()



#fig, ax = plt.subplots()
T = 500 #seconds of simulations, ie iterations/time_steps...
#plotting_colours = iter(plt.cm.rainbow(np.linspace(0,1,T)))

slices = 160 #8km... width...8000/slices = width of sample box...
Xedges = np.linspace(MIN_LON,MAX_LON,slices)
Yedges = np.linspace(MIN_LAT,MAX_LAT,slices)
histo2d_dict = dict()
sum_H = 0
for t in range(0,T):
#    plot_colour = next(plotting_colours)
    lon2 = v2v_exchange_locs_dict[t]['longitude']
    lat2 = v2v_exchange_locs_dict[t]['latitude']
    H, xedges, yedges = np.histogram2d(lon2,lat2, bins=(Xedges,Yedges))

    histo2d_dict[t] = H

    sum_H += H 


#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude')
#ax.set_title('%s' % CITY_NAME)
#plt.show()


total_v2v_exchanges = 0
for i in range(sum_H.shape[0]):
    total_v2v_exchanges += np.sum(sum_H[i])
# note that 3.7 million v2v exchanges in 500 seconds seems a tad much, 7.5k exchanges every second? between 2k taxis... something must be up... investigate further... laterz

#    ax.plot(lon2,lat2,'*k')
#    ax.plot(lon2,lat2, c=plot_colour, marker='+')
#ax.plot(sensor_locations_dict['lon'],sensor_locations_dict['lat'],'ok')
#ax.plot(passenger_trip_start_locs_dict['longitude'], passenger_trip_start_locs_dict['latitude'],'*g')

fig, ax0 = plt.subplots(1,1)
c = ax0.pcolor(sum_H)
ax0.set_title('V2V Exchange Heat-map, %s, total: %i, NUM_TAXIS: %i' % (CITY_NAME, int(total_v2v_exchanges), NUM_TAXIS))
fig.tight_layout()
plt.show()


#comparing heatmaps between 'real' rome traces and taxi-agent traces....

real_taxi_vanet_data_filename = 'No_Pandas_CORRECTED_VANET_rome_combined_29_days.pickle'
real_taxi_data_filepath = '/home/user/ClusterSetUp-All-to-One/Rome-Data/'
with open(real_taxi_data_filepath+real_taxi_vanet_data_filename, 'rb') as handle5:
    real_taxi_data_vanet_dict = pickle.load(handle5)

#real_taxi_vanet_histo2d_dict = dict()
sum_real_taxi_vanet_H2 = 0
# once again, finding the mid-points of the 'real' V2V exchanges....
#minor memory issues... total 'real' data is big... 600mb compressed?
for key, values in real_taxi_data_vanet_dict.items():

    Alon, Alat = np.array(list(zip(*values['Alonglat'])))
    Blon, Blat = np.array(list(zip(*values['Blonglat'])))

    v2v_mid_point_longitude_array = (Alon+Blon)/2
    v2v_mid_point_latitude_array = (Alat+Blat)/2 
    H2, xedges2, yedges2 = np.histogram2d(v2v_mid_point_longitude_array, v2v_mid_point_latitude_array, bins=(Xedges,Yedges))

    sum_real_taxi_vanet_H2 += H2

total_real_v2v_exchanges = 0
for i in range(sum_real_taxi_vanet_H2.shape[0]):
    total_real_v2v_exchanges += np.sum(sum_real_taxi_vanet_H2[i])

fig, ax1 = plt.subplots(1,1)
c = ax1.pcolor(sum_real_taxi_vanet_H2)
ax1.set_title('real... V2V Exchange Heat-map, %s, total: %i, NUM of real TAXIS: 2250' % (CITY_NAME, int(total_real_v2v_exchanges)))
fig.tight_layout()
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
