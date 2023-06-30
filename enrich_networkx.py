"""
This script adds extra metadata to the road layouts in the OSM extract.
In addition to the provided lat, lon coordinates, this looks up elevation
data using DEFRA exports and stores it to an additional 'elevation' attribute.
"""
import json
import pickle

from networkx.readwrite import json_graph
from networkx.exception import NetworkXError
from tqdm import tqdm
from lidar import get_elevation
from maputils import calculate_distance_and_elevation

# Read in the raw JSON representation of the graph
with open("./data/hampshire-latest.json", "r", encoding="utf8") as fobj:
    osm_data = json.load(fobj)

# Convert to networkx, drop the JSON
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

# TODO: Get rid of any dead-end trails

# Calculate elevation change & distance for each edge
for start_id, end_id, data in tqdm(
    osm.edges(data=True), desc="Enriching Edges", leave=False
):
    (
        distance,
        elevation_gain,
        elevation_loss,
    ) = calculate_distance_and_elevation(osm, start_id, end_id)

    data["distance"] = distance
    data["elevation_gain"] = elevation_gain
    data["elevation_loss"] = elevation_loss
    data["via"] = []

    # Clear out any other attributes which aren't needed
    to_remove = [
        attr
        for attr in data
        if attr not in {"distance", "elevation_gain", "elevation_loss", "via"}
    ]
    for attr in to_remove:
        del data[attr]


def get_nodes_to_condense(graph):
    """For the provided graph, find all nodes which have exactly 2 edges
    connected to them. These nodes only provide us with information about the
    geometry of the road, and can be safely removed when calculating the
    best route."""

    nodes = set()
    for id_ in graph.nodes:
        edges = graph.edges(id_)
        node_degree = len(edges)
        if node_degree == 2:
            nodes.add(id_)

    return nodes


def sum_elevation_over_chain(graph, chain, _iters=0):
    """For a chain of 3 nodes, calculate the total elevation gain/loss as
    you traverse the 2 edges joining them. Retain a record of all intermediate
    nodes which are traversed over the course of this route (including the
    one currently being evaluated)."""

    # Fetch the two edges for the chain
    edge_1 = graph[chain[0]][chain[1]]
    edge_2 = graph[chain[1]][chain[2]]

    # Retrieve data for edge 1
    gain_1 = edge_1["elevation_gain"]
    loss_1 = edge_1["elevation_loss"]
    dist_1 = edge_1["distance"]
    via_1 = edge_1.get("via", [])

    # Retrieve data for edge 2
    gain_2 = edge_2["elevation_gain"]
    loss_2 = edge_2["elevation_loss"]
    dist_2 = edge_2["distance"]
    via_2 = edge_2.get("via", [])

    # Calculate whole-chain metrics
    gain = gain_1 + gain_2
    loss = loss_1 + loss_2
    dist = dist_1 + dist_2
    vias = via_1 + [chain[1]] + via_2

    return gain, loss, dist, vias


# Get all of the nodes which can be removed
to_condense = get_nodes_to_condense(osm)
pbar = tqdm(desc="Condensing Graph", total=len(to_condense))
iters = 0
while to_condense:
    # Select a node
    node = to_condense.pop()

    # Find what it connects to, define this route as a chain of 3 nodes
    node_edges = list(osm.edges(node))
    node_chain = [node_edges[0][1], node_edges[0][0], node_edges[1][1]]

    # Add new edge - Start to End
    try:
        gain_se, loss_se, dist_se, vias_se = sum_elevation_over_chain(
            osm, node_chain
        )
        osm.add_edge(
            node_chain[0],
            node_chain[-1],
            via=vias_se,
            elevation_gain=gain_se,
            elevation_loss=loss_se,
            distance=dist_se,
        )
    except KeyError:
        pass  # One way street

    # Add new edge - End to Start
    try:
        gain_es, loss_es, dist_es, vias_es = sum_elevation_over_chain(
            osm, node_chain[::-1]
        )
        osm.add_edge(
            node_chain[-1],
            node_chain[0],
            via=vias_es,
            elevation_gain=gain_es,
            elevation_loss=loss_es,
            distance=dist_es,
        )
    except KeyError:
        pass  # One way street

    # Remove original edges
    for u, v in node_edges:
        try:
            osm.remove_edge(u, v)
        except NetworkXError:
            pass
        try:
            osm.remove_edge(v, u)
        except NetworkXError:
            pass

    # Leave the orphaned node for ease of plotting
    # osm.remove_node(node)

    # Refresh the list of nodes to be condensed
    to_condense = get_nodes_to_condense(osm)
    iters += 1
    pbar.total = len(to_condense) + iters
    pbar.update(1)

pbar.close()

with open("./data/hampshire-latest-compressed.nx", "wb") as fobj:
    pickle.dump(osm, fobj)
