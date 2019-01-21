"""
re-assessing properly line of sight contraint using real taxi trace data... for both cities SF and ROme

"""


import numpy as np
import pickle

import osrm
import polyline

import psycopg2
import pandas.io.sql as pdsql
from sqlalchemy import create_engine


#pointless functions
def HaversineDistance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000
    haversine_distance_metres = m.astype(int)

    return haversine_distance_metres


def HaversineDistanceMatrix(input_gps_pos):

    mat_length = len(input_gps_pos)
    Hdist_matrix = np.zeros((mat_length,mat_length))

    for row in range(0,mat_length):

        for col in range(0,row):
            Hdist = HaversineDistance(input_gps_pos[row][0],input_gps_pos[row][1], input_gps_pos[col][0],input_gps_pos[col][1])
            Hdist_matrix[row,col] = Hdist

#            if (Hdist > min_los_length) & (Hdist < max_los_length):
#                taxis_nolos.append((taxis_Tids[row],taxis_Tids[col],input_gps_pos[row],input_gps_pos[col],Hdist))
                #queck2.append([row,col])

    return Hdist_matrix


def LineOfSightModel(input_gps_pos, taxis_Tids, min_los_length, max_los_length):

    # accpets tupple list of positions: [(long1,lat1),(long2,lat2),... etc.]
    # creates a haversine distance matrix, then finds pairs of taxis that are within
    # min_los_length and max_los_length, typical values [0,100]
    # outputs list of taxi_id pairs and their respective haversine distasnces between them
    # [(taxi_id_A,taxi_id_B,t,haversine_distance)]
    mat_length = len(input_gps_pos)
    Hdist_matrix = np.zeros((mat_length,mat_length),dtype=int)
    taxis_nolos = []
    #queck2 = []
    for row in range(0,mat_length):

        for col in range(0,row):
            Hdist = HaversineDistPC2(input_gps_pos[row],input_gps_pos[col])
            Hdist_matrix[row,col] = Hdist

            if (Hdist > min_los_length) & (Hdist < max_los_length):
                taxis_nolos.append((taxis_Tids[row],taxis_Tids[col],input_gps_pos[row],input_gps_pos[col],Hdist))
                #queck2.append([row,col])


    # Line of Sight Model:
    num_buildings = []
    for i in range(len(taxis_nolos)):
        #i=0

        #longitude,latitude in query
        LoS_execution_str = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxis_nolos[i][2][0]),str(taxis_nolos[i][2][1]),str(taxis_nolos[i][3][0]),str(taxis_nolos[i][3][1])))

        LoS_df = pdsql.read_sql_query(LoS_execution_str,connection)

        num_buildings.append(len(LoS_df))


    return taxis_nolos, num_buildings


### C207

city = 'Rome'
#city = 'SF'

#rome_num_days = 21
#sf_num_days = 4
num_days = 21

max_search_distance = 500 #metres between any two taxis of interest... 



if city is 'Rome':
    # ROME taxi trace data input, currently using 14 days combined (1000ish taxis)
    connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'" 
    psql_connection = psycopg2.connect(connect_str)
    psql_cursor = psql_connection.cursor()
    #rome_taxi_trace_data_path = '/home/user/ClusterSetUp-All-to-One/Rome-Data/No_Pandas_august_edit_taxi_positions_combined_14_days_rome.pickle'
    rome_taxi_trace_data_path = '/home/user/ClusterSetUp-All-to-One/Rome-Data/split_dict_16_till_22_hrs_No_Pandas_august_edit_taxi_positions_combined_21_days_rome.pickle'
    taxi_pos_dict = pickle.load( open( rome_taxi_trace_data_path, 'rb'))


if city is 'SF':
    # SF taxi trace data input, 2 days combined (1000ish taxis)
    connect_str = "dbname ='sfdata_c207' user='postgres' host='localhost' password='postgres'" 
    psql_connection = psycopg2.connect(connect_str)
    psql_cursor = psql_connection.cursor()

    #sf_taxi_trace_data_path = '/home/user/ClusterSetUp-All-to-One/SF-Data/No_Pandas_CORRECTED_sf_taxis_positions_combined_2_days.pickle'
    sf_taxi_trace_data_path = '/home/user/ClusterSetUp-All-to-One/SF-Data/No_Pandas_sf_taxis_positions_combined_4_days.pickle'
    taxi_pos_dict = pickle.load( open( sf_taxi_trace_data_path, 'rb'))

#Outputs
results_data_file_path = '/home/user/Dropbox/RandomDataResults/'
results_filename = ('%s_%i_days_LOS_contraint_investigation_results.pickle' % (city, num_days))



#ROME
#LOS_postgis_query = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxi_a_long),str(taxi_b_lat),str(taxi_b_long),str(taxi_b_lat)))

# SF
#LOS_postgis_query = ("SELECT * FROM sf_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxi_a_long),str(taxi_b_lat),str(taxi_b_long),str(taxi_b_lat)))

"""
# NiGB

connect_str = "dbname ='mike_romedata' user='postgres' host='localhost' password='postgres'" 
psql_connection = psycopg2.connect(connect_str)
psql_cursor = psql_connection.cursor()


#LOS_execution_string = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxis_nolos[i][2][0]),str(taxis_nolos[i][2][1]),str(taxis_nolos[i][3][0]),str(taxis_nolos[i][3][1])))

#(should be x,y, where x=longitude and y=latitude for postgis)

#LOS_execution_string = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxis_nolos[i][2][0]),str(taxis_nolos[i][2][1]),str(taxis_nolos[i][3][0]),str(taxis_nolos[i][3][1])))


psql_cursor.execute(LOS_execution_string)

# ROME taxi trace data input, currently using 14 days combined (1000ish taxis)
rome_taxi_trace_data_path = '/home/elizabeth/MiniTaxiFleets/LineOfSight/No_Pandas_august_edit_taxi_positions_combined_14_days_rome.pickle'
rome_taxi_pos_dict = pickle.load( open( rome_taxi_trace_data_path, 'rb'))

# Rome LOS database query set-up



# SF taxi trace data input, 2 days combined (1000ish taxis)
sf_taxi_trace_data_path = '/home/user/ClusterSetUp-All-to-One/SF-Data/No_Pandas_CORRECTED_sf_taxis_positions_combined_2_days.pickle'
sf_taxi_pos_dict = pickle.load( open( sf_taxi_trace_data_path, 'rb'))
"""



def VectorHaversineDistance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000 #answer is in metres
    return m

def HaversineDistanceArrayFunc2(test_data_set):

    haversine_distance_array = np.zeros((len(test_data_set),len(test_data_set)))
    test_points_array = np.array(list(zip(*test_data_set)))
    dummy_ones_array = np.ones_like(test_points_array[0])

    for j in range(len(test_points_array[0])):

        test_POINT_lat_array = dummy_ones_array*test_points_array[1][j]
        test_POINT_long_array = dummy_ones_array*test_points_array[0][j]

        haversine_distance_array[j][j:] = VectorHaversineDistance(test_points_array[0][j:],test_points_array[1][j:],test_POINT_long_array[j:],test_POINT_lat_array[j:])
    return haversine_distance_array


nolos_dist_list = []
los_dist_list = []

num_timestamps = len(list(taxi_pos_dict.keys()))

processed_keys = []

#for key, values in rome_taxi_pos_dict.items():
for key, values in taxi_pos_dict.items():
    # Eucledean distance matrix between all taxis
    # Select those less than 500m?

    raw_taxi_positions_list = values['longlat']

    #taxi_haversine_dist_array = HaversineDistanceMatrix(raw_taxi_positions_list)

    taxi_haversine_dist_array = HaversineDistanceArrayFunc2(raw_taxi_positions_list)

    taxi_pairs_within_range_array = np.where((taxi_haversine_dist_array>0) & (taxi_haversine_dist_array<max_search_distance))

    # evaluate LOS condition for all communicating taxi pairs
    for i in range(len(taxi_pairs_within_range_array)):

        taxi_a_lat = raw_taxi_positions_list[taxi_pairs_within_range_array[0][i]][1]
        taxi_a_long = raw_taxi_positions_list[taxi_pairs_within_range_array[0][i]][0]

        taxi_b_lat = raw_taxi_positions_list[taxi_pairs_within_range_array[1][i]][1]
        taxi_b_long = raw_taxi_positions_list[taxi_pairs_within_range_array[1][i]][0]
     

        #ROME
        if city is 'Rome':
            LOS_postgis_query = ("SELECT * FROM rome_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxi_a_long),str(taxi_b_lat),str(taxi_b_long),str(taxi_b_lat)))

        elif city is 'SF':
        # SF
            LOS_postgis_query = ("SELECT * FROM sf_buildings WHERE ST_Intersects(ST_SetSRID('LINESTRING (%s %s, %s %s)'::geometry,4326), geom);" % (str(taxi_a_long),str(taxi_b_lat),str(taxi_b_long),str(taxi_b_lat)))

        psql_cursor.execute(LOS_postgis_query)
        los_result = psql_cursor.fetchone()

        if los_result is None:
            los_dist_list.append(int(taxi_haversine_dist_array[taxi_pairs_within_range_array[0][i]][taxi_pairs_within_range_array[1][i]]))

        else:
            nolos_dist_list.append(int(taxi_haversine_dist_array[taxi_pairs_within_range_array[0][i]][taxi_pairs_within_range_array[1][i]]))

    ####    # evaluate Routing distance between all communicating taxi pairs?



    print(key)
    processed_keys.append(key)
    print(len(processed_keys)/num_timestamps)

psql_cursor.close()


results_dict = {'nolos':nolos_dist_list, 'los':los_dist_list}

#saving results data
#results_data_file_path = '/home/user/Dropbox/RandomDataResults/'
#results_filename = 'SF_2_days_LOS_contraint_investigation_results.pickle'
with open((results_data_file_path+results_filename),'wb') as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



"""
# now a separate script... 
##### test plots.....

import matplotlib.pyplot as plt
import pickle
import numpy as np

sf_taxi_los_results_path = '/home/user/Dropbox/RandomDataResults/' + 'SF_2_days_LOS_contraint_investigation_results.pickle'

sf_taxi_los_results_dict = pickle.load( open( sf_taxi_los_results_path, 'rb'))


histo_bins = [0,25,50,100,150,200,250,300,350]

los_results_histo = np.histogram(sf_taxi_los_results_dict['los'],bins = histo_bins, density=True)
nolos_results_histo = np.histogram(sf_taxi_los_results_dict['nolos'], bins = histo_bins, density=True)

plotting_bins = np.cumsum(np.diff(los_results_histo[1]))

plt.plot(plotting_bins,los_results_histo[0],'or',plotting_bins,nolos_results_histo[0],'sg')
plt.xlabel('Distance/[m]')
plt.ylabel('Some form of Frequency...?')
plt.show()







test_data_set = rome_taxi_pos_dict[49150]['longlat'] 
#test_data2 = list(zip(*test_data_set))

haversine_distance_array = np.zeros((len(test_data_set),len(test_data_set)))
test_points_array = np.array(list(zip(*test_data_set)))
dummy_ones_array = np.ones_like(test_points_array[0])

j = 0

test_POINT_lat_array = dummy_ones_array*test_points_array[1][j]
test_POINT_long_array = dummy_ones_array*test_points_array[0][j]

queck = VectorHaversineDistance(test_points_array[0],test_points_array[1],test_POINT_long_array,test_POINT_lat_array)

haversine_distance_array[j][j:] = queck


"""





