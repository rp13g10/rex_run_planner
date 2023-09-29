"""Enables the splitting of a graph into multiple subgraphs, and the
regeneration of the initial graph after processing."""
import math
import os
from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np
from networkx import Graph, compose

from rex_run_planner.containers import BBox


class GraphSplitter:
    """Class which can split a graph into a grid of subgraphs according to
    lat/lon, and recreate the original graph from these subgraphs. Useful
    if you want to perform parallel processing."""

    def __init__(self, graph: Graph):
        """Create a new GraphSplitter instance for the provided graph

        Args:
            graph (Graph): The graph to be split. Nodes must have been tagged
              with 'lat' and 'lon' attributes.
        """

        self.graph = graph
        self.subgraph_nodes = defaultdict(list)
        self.subgraphs = {}
        self.cross_boundary_edges = {}

        # Set subgraph size according to core count, this may need to be
        # adjusted for very large graphs if memory usage proves too high
        if no_cpus := os.cpu_count():
            self.no_subgraphs = no_cpus
        else:
            self.no_subgraphs = math.ceil(len(self.graph.nodes) / 10000)
        self.grid_size = math.ceil(math.sqrt(self.no_subgraphs))
        self.grid = {}

    def get_edge_nodes(self) -> Set[int]:
        """Based on the edges which cross the boundaries between grid squares,
        fetch a set of nodes which are on the edge of a grid square.

        Returns:
            Set[int]: The set of nodes which are on the edge of a grid square
        """
        nodes = set()
        for start, end in self.cross_boundary_edges:
            nodes.add(start)
            nodes.add(end)
        return nodes

    edge_nodes = property(fget=get_edge_nodes)

    def fetch_node_coords(self, node_id: int) -> Tuple[int, int]:
        """Convenience function, retrieves the latitude and longitude for a
        single node in a graph."""
        node = self.graph.nodes[node_id]
        lat = node["lat"]
        lon = node["lon"]
        return lat, lon

    def _get_full_graph_bbox(self) -> BBox:
        """Generate the bounding box which encompasses the entire graph

        Returns:
            BBox: Contains the min & max latitudes & longitudes for the
              internal graph
        """
        min_lat = None
        min_lon = None
        max_lat = None
        max_lon = None
        for _, node_data in self.graph.nodes(data=True):
            node_lat = node_data["lat"]
            node_lon = node_data["lon"]
            if min_lat is None:
                min_lat = node_lat
                min_lon = node_lon
                max_lat = node_lat
                max_lon = node_lon
            else:
                min_lat = min(min_lat, node_lat)  # type: ignore
                min_lon = min(min_lon, node_lon)  # type: ignore
                max_lat = max(max_lat, node_lat)  # type: ignore
                max_lon = max(max_lon, node_lon)  # type: ignore

        bbox = BBox(
            min_lat=min_lat,  # type: ignore
            min_lon=min_lon,  # type: ignore
            max_lat=max_lat,  # type: ignore
            max_lon=max_lon,  # type: ignore
        )

        return bbox

    def _generate_grid(self, bbox: BBox) -> Dict[Tuple[int, int], BBox]:
        """Subdivide a bounding box into a grid of smaller bounding boxes

        Args:
            bbox (BBox): The bounding box for all candidate routes

        Returns:
            Dict[Tuple[int, int], BBox]: A dictionary mapping grid identifiers
              (row & column numbers) to bounding boxes
        """

        lat_bins = np.linspace(bbox.min_lat, bbox.max_lat, self.grid_size + 1)

        lon_bins = np.linspace(bbox.min_lon, bbox.max_lon, self.grid_size + 1)

        last_lat = None
        last_lon = None
        grid = {}
        for lat_inx, lat in enumerate(lat_bins):
            if lat_inx == 0:
                last_lat = lat
                continue

            for lon_inx, lon in enumerate(lon_bins):
                if lon_inx == 0:
                    last_lon = lon
                    continue

                bbox = BBox(
                    min_lat=last_lat,  # type: ignore
                    min_lon=last_lon,  # type: ignore
                    max_lat=lat,
                    max_lon=lon,
                )

                grid[(lat_inx, lon_inx)] = bbox

                last_lon = lon

            last_lat = lat

        return grid

    def _get_grid_square_for_node(
        self, node_id: int, tagged_grid: Dict[Tuple[int, int], BBox]
    ) -> Tuple[int, int]:
        """For a given node_id and grid of bounding boxes, calculate which grid
        square the node should be placed into

        Args:
            node_id (int): The node to be assigned a grid square
            tagged_grid (Dict[Tuple[int, int], BBox]): A dictionary mapping
              grid square IDs to bounding boxes

        Raises:
            KeyError: If a grid square for the node cannot be found, an error
              will be raised.

        Returns:
            Tuple[int, int]: The grid square ID for the provided node_id
        """

        node_lat, node_lon = self.fetch_node_coords(node_id)

        excluded_lats = set()
        for (lat_inx, lon_inx), bbox in tagged_grid.items():
            # Find the correct column
            if lat_inx in excluded_lats:
                continue
            min_lat = bbox.min_lat
            max_lat = bbox.max_lat
            if min_lat <= node_lat <= max_lat:
                # Find the correct row
                min_lon = bbox.min_lon
                max_lon = bbox.max_lon
                if min_lon <= node_lon <= max_lon:
                    self.subgraph_nodes[(lat_inx, lon_inx)].append(node_id)
                    return (lat_inx, lon_inx)
            else:
                excluded_lats.add(lat_inx)
                continue

        raise KeyError(f"Unable to locate a grid square for node {node_id}")

    def _tag_nodes_with_grid_squares(self):
        """For every node in the internal grid, work out which grid square
        it should be assigned to"""

        full_bbox = self._get_full_graph_bbox()
        tagged_grid = self._generate_grid(full_bbox)
        self.grid = tagged_grid

        for node_id in self.graph.nodes:
            grid_square = self._get_grid_square_for_node(node_id, tagged_grid)
            self.graph.nodes[node_id]["grid_square"] = grid_square

    def _get_all_cross_boundary_edges(self) -> Dict[Tuple[int, int], Dict]:
        """Identify every edge in the internal graph which starts in one grid
        square and ends in another

        Returns:
            Dict[Tuple[int, int], Dict]: A dictionary where each key represents
              the start & end point of an edge, and the corresponding value
              represents any data attributes assigned to it
        """

        cb_edges = {}
        for start_node, end_node in self.graph.edges:
            start_square = self.graph.nodes[start_node]["grid_square"]
            end_square = self.graph.nodes[end_node]["grid_square"]

            if start_square != end_square:
                edge_attrs = self.graph.get_edge_data(start_node, end_node)
                cb_edges[(start_node, end_node)] = edge_attrs

        return cb_edges

    @staticmethod
    def _clear_grid_attributes(graph: Graph) -> Graph:
        """Remove any references to grid squares from the provided graph

        Args:
            graph (Graph): The graph to be processed

        Returns:
            Graph: A copy of the provided graph with any 'grid_square' node
              attributes removed
        """
        for node_id in graph.nodes:
            try:
                del graph.nodes[node_id]["grid_square"]
            except KeyError:
                pass

        return graph

    def explode_graph(self):
        """Split the internal graph into subgraphs. Calling this function will
        clear the internal 'self.graph' variable, and populate the
        'self.subgraphs' variable.
        """

        # Assign nodes to grid squares
        self._tag_nodes_with_grid_squares()

        # Record the details of any edges which span more than one grid square
        cb_edges = self._get_all_cross_boundary_edges()
        self.cross_boundary_edges = cb_edges

        # Split the graph into subgraphs
        for grid_square, nodes in self.subgraph_nodes.items():
            subgraph = self.graph.subgraph(nodes).copy()
            self.subgraphs[grid_square] = subgraph

        # Delete the original graph to free up memory
        del self.graph

    def rebuild_graph(self):
        """Rebuild the original graph from the internal subgraphs. Calling this
        function will repopulate the internal 'self.graph' variable, and clear
        the 'self.subgraphs' variable.
        """

        # Combine all subgraphs into a single graph
        full_graph = None  # type: ignore
        for subgraph in self.subgraphs.values():
            if full_graph is None:
                full_graph = subgraph
            full_graph: Graph = compose(full_graph, subgraph)

        # Add back in the edges which crossed grid squares
        for (start, end), edge_attrs in self.cross_boundary_edges.items():
            full_graph.add_edge(start, end, **edge_attrs)

        # Remove all references to grid squares to free up memory
        full_graph = self._clear_grid_attributes(full_graph)
        self.graph = full_graph
        self.grid = {}
        del self.subgraph_nodes
        del self.subgraphs
        del self.cross_boundary_edges
