import math
import networkx
from geopy.distance import distance


def find_nearest_node(graph, lat, lon):
    """For a given latitude/longitude, find the node which is geographically
    closest to it, that is connected to at least one other node.
    As the network has already been compressed, this node will always be at
    a junction."""
    isolates = set(networkx.isolates(graph))
    closest_node = None
    shortest_dist = None
    for id_, attrs in graph.nodes.items():
        # Ignore orphaned nodes
        if id_ in isolates:
            continue

        node_lat = attrs["lat"]
        node_lon = attrs["lon"]

        close_lat = math.isclose(lat, node_lat, abs_tol=0.001)

        if not close_lat:
            continue

        close_lon = math.isclose(lon, node_lon, abs_tol=0.001)

        if not close_lon:
            continue

        # node_dist = math.dist([lat, lon], [node_lat, node_lon])
        node_dist = distance((lat, lon), (node_lat, node_lon))

        if not closest_node:
            closest_node = id_
            shortest_dist = node_dist
        else:
            if shortest_dist > node_dist:  # type: ignore
                closest_node = id_
                shortest_dist = node_dist

    if not closest_node:
        raise ValueError(
            f"No nearby nodes were found for {lat:.4f}, {lon:.4f}"
        )

    return closest_node
