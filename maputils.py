import networkx
import math
from geopy.distance import distance
import numpy as np

from lidar import get_elevation


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

        close_lat = math.isclose(lat, node_lat, abs_tol=0.0005)

        if not close_lat:
            continue

        close_lon = math.isclose(lon, node_lon, abs_tol=0.0005)

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


def fetch_node_coords(graph, node_id):
    """Convenience function, retrieves the latitude and longitude for a
    single node in a graph."""
    node = graph.nodes[node_id]
    lat = node["lat"]
    lon = node["lon"]
    return lat, lon


def calculate_distance_and_elevation(graph, start_id, end_id, mode="metric"):
    """For a connected pair of nodes in the provided graph, estimate the
    change in elevation by fetching the altitude once every 10 metres and
    summing the loss/gain between steps. In theory, this should be more
    accurate than simply assuming that the edge follows a consistent gradient
    along its length."""

    # Fetch lat/lon for the start/end nodes
    start_lat, start_lon = fetch_node_coords(graph, start_id)
    end_lat, end_lon = fetch_node_coords(graph, end_id)

    # Calculate distance from A to B
    dist_change = distance((start_lat, start_lon), (end_lat, end_lon))

    # Calculate number of checks required to get elevation every 10 metres
    dist_change_m = dist_change.meters
    no_checks = math.ceil(dist_change_m / 10)

    # Generate latitudes & longitudes for each checkpoint
    lat_checkpoints = np.linspace(start_lat, end_lat, no_checks)
    lon_checkpoints = np.linspace(start_lon, end_lon, no_checks)

    # Calculate elevation at each checkpoint
    elevations = []
    for lat, lon in zip(lat_checkpoints, lon_checkpoints):
        elevation = get_elevation(lat, lon)
        elevations.append(elevation)

    # Work out the sum of elevation gains/losses between checkpoints
    last_elevation = None
    elevation_gain = 0.0
    elevation_loss = 0.0
    for elevation in elevations:
        if not last_elevation:
            last_elevation = elevation
            continue
        if elevation > last_elevation:
            elevation_gain += elevation - last_elevation
        elif elevation < last_elevation:
            elevation_loss += last_elevation - elevation
        last_elevation = elevation

    # Retrieve distance in the desired form
    if mode == "metric":
        dist_change = dist_change.kilometers
    elif mode == "imperial":
        dist_change = dist_change.miles
    else:
        raise ValueError(
            f'mode must be one of "metric", "imperial". Got "{mode}"'
        )

    return dist_change, elevation_gain, elevation_loss
