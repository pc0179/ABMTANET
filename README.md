# ABMTANET
Agent-based modelling of taxi ad-hoc networks

Taxi trips are generated following differing initial conditions and demand models. A simulation is then run using varying number of taxis/agents to provide passenger transport as well as connectivity to sensors distributed across simulated urban environment.

A parameter, called Alpha is used to varying the road network/graph edge weights. Edge weights are varied depending on proximity to sensors (i.e. minimum distance between edge mid-point and sensor location). Alpha equal to zero would imply all routes that passengers take (inside an agent/taxi) are routed according to the shortest travel time (or road trip length distance) thus ignoring any sensors along the route. Alpha equal to 1 would imply all routes are routed along edges that contain (or are within V2X exchange proximity) of sensors. A limit of twice the shortest route length was used alongside a strict prohibition of ciruclar routes (i.e. visiting edges more than twice during a passenger trip service).

Varying passenger demand was acheived by selecting areas of the simulated urban environement to act as sinks or sources. For example, a typical morning rush-hour commute can be simuated by geo-fencing the city's central business district and forcing trips to end their (whilst allowing the start location of passenger trips elsewhere in the simualted environment).

