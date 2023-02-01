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

# NOTE - Traversal
# Find starting node closest to selected point
# Travel to each connected node, keeping a tally of distance & elevation
# Track visited edges to avoid repeats
# Track maximal elevation ratio(?) to each point

from maputils import find_nearest_node, calculate_distance_and_elevation
from enrich_networkx import osm

# Start by nearby park
start_lat = 50.969540
start_lon = -1.383318

# Enter network at nearest available point
start_node = find_nearest_node(osm, start_lat, start_lon)

# Start at provided node
# For each neighbour:
# calculate elevation/distance changes when moving to this node
# record these candidate moves in a list of edges, storing any required
#   stats

# For each candidate in candidates:
# find neighbours for this node which have not yet been traversed
# for each neighbour:
# calculate elevation/distance changes
# append this move to the current candidate
# check against early stopping conditions, break if any are met
# if neighbour is not the start node
# append the current candidate to a new_candidates
# if neighbour is the start node and total distance is suitable
# append the current candidate to final_candidates
# set candidates = new_candidates on final iteration
