"""
https://stackoverflow.com/questions/41920398/valueerror-unsupported-pickle-protocol-4-with-pandas
https://networkx.github.io/documentation/stable/_modules/networkx/readwrite/gpickle.html#read_gpickle

"""

import osmnx
import numpy as np
import networkx as nx

#G =  osmnx.graph_from_file('/home/user/MiniTaxiFleets/AnuStart/mini_centraal_sac_town.osm', network_type='drive',simplify=True) 

#G =  osmnx.graph_from_file('/home/user/MiniTaxiFleets/road_network_osm_files/cologne_centraal_ish_8km2.osm', network_type='drive',simplify=True) 

#all on c207
path_to_osm_files = '/home/user/Downloads/osm_city_road_maps/'

SF_central_filename = 'SF_central_64km2_-122.479,37.733_-122.388,37.805.osm'
Roma_centrale_filename = 'roma_centrale_64km2_12.442,41.856_12.5387,41.928.osm'
Birmingham_central_filename = 'Birmingham_central_64km2_-1.971,52.447_-1.825,52.508.osm'

G1 =  osmnx.graph_from_file(path_to_osm_files+SF_central_filename, network_type='drive',simplify=True) 

G2 = osmnx.graph_from_file(path_to_osm_files+Roma_centrale_filename, network_type='drive',simplify=True) 

G3 = osmnx.graph_from_file(path_to_osm_files+Birmingham_central_filename, network_type='drive',simplify=True) 


G1b = osmnx.core.add_edge_lengths(G1)
G2b = osmnx.core.add_edge_lengths(G2)
G3b = osmnx.core.add_edge_lengths(G3)

#nx.write_gpickle(G2, "Cologne_Centraal_Road_Network.gpickle", protocol=2)

nx.write_gpickle(G1b, "Highest_Protocol_SF_Central_Road_Network.gpickle", protocol=-100)

nx.write_gpickle(G2b, "Highest_Protocol_Roma_Centrale_Road_Network.gpickle", protocol=-100)

nx.write_gpickle(G3b, "Highest_Protocol_Birmingham_central_Road_Network.gpickle", protocol=-100)
