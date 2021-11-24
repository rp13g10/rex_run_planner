# NOTE - Proposed Algorithm
# Load hampshire-latest.pypgr (https://github.com/AndGem/OsmToRoadGraph)
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