"""Contains the GraphEnricher class, which will be executed if this script
is executed directly."""

import json
import pickle
from itertools import repeat
from multiprocessing.pool import Pool
from typing import List, Tuple, Union, Optional

import networkx as nx
from tqdm import tqdm
from networkx import Graph
from networkx.exception import NetworkXError
from networkx.readwrite import json_graph

from rex_run_planner.containers import RouteConfig, ChainMetrics
from rex_run_planner.data_prep.lidar import get_elevation
from rex_run_planner.data_prep.graph_utils import GraphUtils
from rex_run_planner.data_prep.graph_splitter import GraphSplitter
from rex_run_planner.data_prep.graph_condenser import condense_graph

# TODO: Implement parallel processing for condensing of enriched graphs.
#       Subdivide graph into grid, distribute condensing of each subgraph
#       then stitch the results back together. Final pass will be required to
#       process any edges which were temporarily removed as they bridged
#       multiple subgraphs.


class GraphEnricher(GraphUtils):
    """Class which enriches the data which is provided by Open Street Maps.
    Unused data is stripped out, and elevation data is added for both nodes and
    edges. The graph itself is condensed, with nodes that lead to dead ends
    or only represent a bend in the route being removed.
    """

    def __init__(
        self,
        source_path: str,
        config: RouteConfig,
    ):
        """Create an instance of the graph enricher class based on the
        contents of the networkx graph specified by `source_path`

        Args:
            source_path (str): The location of the networkx graph to be
              enriched. The graph must have been saved to json format.
            dist_mode (str, optional): The preferred output mode for distances
              which are saved to node edges. Returns kilometers if set to
              metric, miles if set to imperial. Defaults to "metric".
            elevation_interval (int, optional): When calculating elevation
              changes across an edge, values will be estimated by taking
              checkpoints at regular checkpoints. Smaller values will result in
              more accurate elevation data, but may slow down the calculation.
              Defaults to 10.
            max_condense_passes (int, optional): When condensing the graph, new
              dead ends may be created on each pass (i.e. if one dead end
              splits into two, pass 1 removes the 2 dead ends, pass 2 removes
              the one they split from). Use this to set the maximum number of
              passes which will be performed.
        """

        # Store down user preferences
        msg = (
            'mode must be one of "metric", "imperial". '
            f'Got "{config.dist_mode}"'
        )
        assert config.dist_mode in {
            "metric",
            "imperial",
        }, msg
        self.config = config

        # Read in the contents of the graph
        graph = self.load_graph(source_path)

        # Store down core attributes
        super().__init__(graph, config)

        # Create container objects
        self.nodes_to_condense = []
        self.nodes_to_remove = []

    def load_graph(self, source_path: str) -> Graph:
        """Read in the contents of the JSON file specified by `source_path`
        to a networkx graph.

        Args:
            source_path (str): The location of the networkx graph to be
              enriched. The graph must have been saved to json format.

        Returns:
            Graph: A networkx graph with the contents of the provided json
              file.
        """

        # Read in the contents of the JSON file
        with open(source_path, "r", encoding="utf8") as fobj:
            graph_data = json.load(fobj)

        # Convert it back to a networkx graph
        graph = json_graph.adjacency_graph(graph_data)

        return graph

    def _enrich_source_nodes(self):
        """For each node in the graph, attempt to fetch elevation info from
        the loaded LIDAR data. If no elevation information is available, the
        node will be dropped to minimise memory usage."""

        to_delete = set()
        for inx, attrs in tqdm(
            self.graph.nodes.items(), desc="Enriching Nodes", leave=False
        ):
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
                self.graph.nodes[inx]["elevation"] = elevation

        # Remove nodes with no elevation data
        self.graph.remove_nodes_from(to_delete)

    def _enrich_source_edges(self):
        """For each edge in the graph, estimate the distance, elevation gain
        and elevation loss when traversing it. Strip out all other metadata
        to minimise the memory footprint of the graph.
        """

        # Calculate elevation change & distance for each edge
        for start_id, end_id, data in tqdm(
            self.graph.edges(data=True), desc="Enriching Edges", leave=True
        ):
            (
                dist_change,
                elevation_gain,
                elevation_loss,
            ) = self._estimate_distance_and_elevation_change(start_id, end_id)

            data["distance"] = dist_change
            data["elevation_gain"] = elevation_gain
            data["elevation_loss"] = elevation_loss
            data["via"] = []

            # Clear out any other attributes which aren't needed
            to_remove = [
                attr
                for attr in data
                if attr
                not in {"distance", "elevation_gain", "elevation_loss", "via"}
            ]
            for attr in to_remove:
                del data[attr]

    def enrich_graph(
        self,
        full_target_loc: Optional[str] = None,
        cond_target_loc: Optional[str] = None,
    ):
        """Enrich the graph with elevation data, calculate the change in
        elevation for each edge in the graph, and shrink the graph as much
        as possible.

        Args:
            full_target_loc (Optional[str]): The location which the full graph
              should be saved to (pre-compression). This will be helpful if you
              intend on plotting any routes afterwards, as not all nodes will
              be present in the compressed graph. Defaults to None.
            cond_target_loc (Optional[str]): The loation which the condensed
              graph should be saved to. Defaults to None.
        """
        self._enrich_source_nodes()
        self._enrich_source_edges()

        if full_target_loc:
            self.save_graph(full_target_loc)

        self.graph = condense_graph(self.graph)

        if cond_target_loc:
            self.save_graph(cond_target_loc)

    def save_graph(self, target_loc: str):
        """Once processed, pickle the graph to the local filesystem ready for
        future use.

        Args:
            target_loc (str): The location which the graph should be saved to.
        """
        with open(target_loc, "wb") as fobj:
            pickle.dump(self.graph, fobj)
