"""
This script adds extra metadata to the road layouts in the OSM extract.
In addition to the provided lat, lon coordinates, this looks up elevation
data using DEFRA exports and stores it to an additional 'elevation' attribute.
"""
import json

from networkx.readwrite import json_graph
from tqdm import tqdm
from lidar import get_elevation
from maputils import calculate_distance_and_elevation, find_nearest_node

with open("./data/hampshire-latest.json", "r", encoding="utf8") as fobj:
    osm_data = json.load(fobj)

osm = json_graph.adjacency_graph(osm_data)
del osm_data

# Add elevation to all nodes where it's available
to_delete = set()
for inx, attrs in tqdm(osm.nodes.items(), desc="Enriching Nodes", leave=False):

    # Unpack coordinates
    lat = attrs["lat"]
    lon = attrs["lon"]

    # Fetch elevation
    try:
        elevation = get_elevation(lat, lon)
    except FileNotFoundError:
        elevation = None

    if not elevation:
        # Mark node for deletion
        to_delete.add(inx)
    else:
        # Add elevation to node
        osm.nodes[inx]["elevation"] = elevation

# Remove nodes with no elevation data
osm.remove_nodes_from(to_delete)

# Calculate elevation change & distance for each edge
for start_id, end_id in tqdm(osm.edges, desc="Enriching Edges", leave=False):
    distance, elevation = calculate_distance_and_elevation(
        osm, start_id, end_id
    )

    osm[start_id][end_id]["distance"] = distance
    osm[start_id][end_id]["elevation"] = elevation
