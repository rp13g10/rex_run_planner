"""Contains the GraphEnricher class, which will be executed if this script
is executed directly."""

import math
from abc import ABC
from typing import List, Tuple

import numpy as np
from geopy.distance import distance
from networkx import Graph

from rex_run_planner.containers import RouteConfig
from rex_run_planner.data_prep.lidar import get_elevation

# TODO: Implement parallel processing for condensing of enriched graphs.


class GraphUtils(ABC):
    """Contains utility functions which are to be shared across both the
    GraphEnricher and GraphTagger classes.
    """

    def __init__(self, graph: Graph, config: RouteConfig):
        self.graph = graph
        self.config = config

    def fetch_node_coords(self, node_id: int) -> Tuple[int, int]:
        """Convenience function, retrieves the latitude and longitude for a
        single node in a graph."""
        node = self.graph.nodes[node_id]
        lat = node["lat"]
        lon = node["lon"]
        return lat, lon

    def _get_elevation_checkpoints(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> Tuple[List[float], List[float], distance]:
        """Given a start & end point, return a list of equally spaced latitudes
        and longitudes between them. These can then be used to estimate the
        elevation change between start & end by calculating loss/gain between
        each checkpoint.

        Args:
            start_lat (float): Latitude for the start point
            start_lon (float): Longitude for the start point
            end_lat (float): Latitude for the end point
            end_lon (float): Longitude for the end point

        Returns:
            Tuple[List[float], List[float], distance]: A list of latitudes and
              a corresponding list of longitudes which represent points on an
              edge of the graph. A geopy distance object which shows the total
              distance between the start & end point
        """
        # Calculate distance from A to B
        dist_change = distance((start_lat, start_lon), (end_lat, end_lon))

        # Calculate number of checks required to get elevation every N metres
        dist_change_m = dist_change.meters
        no_checks = math.ceil(dist_change_m / self.config.elevation_interval)
        no_checks = max(2, no_checks)

        # Generate latitudes & longitudes for each checkpoint
        lat_checkpoints = list(np.linspace(start_lat, end_lat, no_checks))
        lon_checkpoints = list(np.linspace(start_lon, end_lon, no_checks))

        return lat_checkpoints, lon_checkpoints, dist_change

    def _calculate_elevation_change_for_checkpoints(
        self, lat_checkpoints: List[float], lon_checkpoints: List[float]
    ) -> Tuple[float, float]:
        """For the provided latitude/longitude coordinates, estimate the total
        elevation gain/loss along the entire route.

        Args:
            lat_checkpoints (List[float]): A list of equally spaced latitudes
              which represent points on an edge of the graph
            lon_checkpoints (List[float]): A list of equally spaced longitudes
              which represent points on an edge of the graph

        Returns:
            Tuple[float, float]: Elevation gain in metres, elevation loss in
              metres
        """
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

        return elevation_gain, elevation_loss

    def _estimate_distance_and_elevation_change(
        self, start_id: int, end_id: int
    ) -> Tuple[float, float, float]:
        """For a given start & end node, estimate the change in elevation when
        traversing the edge between them. The number of samples used to
        estimate the change in elevation is determined by the
        self.elevation_interval attribute.

        Args:
            start_id (int): The starting node for edge traversal
            end_id (int): The end node for edge traversal

        Returns:
            Tuple[float, float, float]: The distance change, elevation gain
              and elevation loss
        """
        # Fetch lat/lon for the start/end nodes
        start_lat, start_lon = self.fetch_node_coords(start_id)
        end_lat, end_lon = self.fetch_node_coords(end_id)

        (
            lat_checkpoints,
            lon_checkpoints,
            dist_change,
        ) = self._get_elevation_checkpoints(
            start_lat, start_lon, end_lat, end_lon
        )

        (
            elevation_gain,
            elevation_loss,
        ) = self._calculate_elevation_change_for_checkpoints(
            lat_checkpoints, lon_checkpoints
        )

        # Retrieve distance in the desired form
        if self.config.dist_mode == "metric":
            dist_change = dist_change.kilometers
        else:
            dist_change = dist_change.miles

        return dist_change, elevation_gain, elevation_loss

    def _get_straight_line_distance_and_elevation_change(
        self, start_id: int, end_id: int
    ) -> Tuple[float, float, float]:
        """Calculate the change in elevation accrued when travelling in a
        straight line from one node to another

        Args:
            start_id (int): The ID of the start node
            end_id (int): The ID of the end node

        Returns:
            Tuple[float, float, float]: The distance from start_id to end_id,
              the elevation gain and the elevation loss
        """

        # Elevation change
        start_ele = self.graph.nodes[start_id]["elevation"]
        end_ele = self.graph.nodes[end_id]["elevation"]
        change = end_ele - start_ele
        gain = max(0, change)
        loss = abs(min(0, change))

        # Distance change
        start_lat, start_lon = self.fetch_node_coords(start_id)
        end_lat, end_lon = self.fetch_node_coords(end_id)
        dist = distance((start_lat, start_lon), (end_lat, end_lon))
        if self.config.dist_mode == "metric":
            dist = dist.kilometers
        else:
            dist = dist.miles

        return dist, gain, loss
