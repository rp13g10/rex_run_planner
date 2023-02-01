import math
from geopy.distance import distance


def find_nearest_node(graph, lat, lon):
    closest_node = None
    shortest_dist = None
    for inx, attrs in graph.nodes.items():
        node_lat = attrs["lat"]
        node_lon = attrs["lon"]

        close_lat = math.isclose(lat, node_lat, abs_tol=0.0002)

        if not close_lat:
            continue

        close_lon = math.isclose(lon, node_lon, abs_tol=0.0002)

        if not close_lon:
            continue

        # node_dist = math.dist([lat, lon], [node_lat, node_lon])
        node_dist = distance((lat, lon), (node_lat, node_lon))

        if not closest_node:
            closest_node = inx
            shortest_dist = node_dist
        else:
            if shortest_dist > node_dist:  # type: ignore
                closest_node = inx
                shortest_dist = node_dist

    if not closest_node:
        raise ValueError(
            f"No nearby nodes were found for {lat:.4f}, {lon:.4f}"
        )

    return closest_node


def calculate_distance_and_elevation(graph, start_id, end_id):

    start_node = graph.nodes[start_id]
    end_node = graph.nodes[end_id]

    start_lat = start_node["lat"]
    start_lon = start_node["lon"]
    start_ele = start_node["elevation"]

    end_lat = end_node["lat"]
    end_lon = end_node["lon"]
    end_ele = end_node["elevation"]

    ele_change = end_ele - start_ele
    dist_change = distance((start_lat, start_lon), (end_lat, end_lon))

    return dist_change, ele_change
