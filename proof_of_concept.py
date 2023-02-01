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
