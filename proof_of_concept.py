# NOTE - Data Sources
# https://environment.data.gov.uk/DefraDataDownload/?Mode=survey - LIDAR Data (elevation)
# https://www.openstreetmap.org/export - Map Data
# For seed data, used https://download.geofabrik.de/europe/great-britain/england/hampshire.html

# NOTE - Tools
# https://github.com/AndGem/OsmToRoadGraph - Extract roads from OSM data

# NOTE - Proposed Algorithm

# Read nodes & edges into a networkx graph
# Generalize elevation getter to use multiple lidar files
# Annotate each edge with an elevation gain & loss
#   Take start & end point
#   Generate list of points to form a straight line from start to end
#   Calculate elevation at each point
#   Calculate total gain/loss
# Implement pathfinding algorithm to maximize elevation
# Export the route to .gpx

# TODO
# Improve pruning algorithm
# - Use proximity to local maxima / stdev of elevation in certain radius?
# - Minimise memory footprint of network, strip out extra data & remove any
#   dead ends?
# Check whether OSM data already includes elevation info?
# See whether multithreading is an option

from maputils import find_nearest_node, calculate_distance_and_elevation
from enrich_networkx import osm

# Start by nearby park
start_lat = 50.969540
start_lon = -1.383318

# Finish at the top of Pitmore Hill
end_lat = 50.998649
end_lon = -1.352969

# Enter network at nearest available point
start_node = find_nearest_node(osm, start_lat, start_lon)
end_node = find_nearest_node(osm, end_lat, end_lon)

# Calculate distance from the park to the top of the hill
distance, elevation = calculate_distance_and_elevation(
    osm, start_node, end_node, mode="imperial"
)

# Performance improvements:
# Add distance/elevation changes to node edges
# Prune similar routes?
# DOWNLOAD MORE LIDAR DATA!

# sudo docker run -v /myredis/conf:/home/ross/repos/rex-run-planner/redis --name myredis -p 6379:6379 redis redis-server /home/ross/repos/rex-run-planner/redis/redis.conf

# 5km Performance
# Stratified pruning to 100k: 1:25:00, 72,044 routes found

# 10km Performance
# Limit to 100k best after each iteration: 20:32, no routes found
