"""Contains the GraphEnricher class, which will be executed if this script
is executed directly."""

import json
import pickle
from typing import List, Tuple, Union, Optional

import networkx as nx
from tqdm import tqdm
from networkx.exception import NetworkXError
from networkx.readwrite import json_graph

from rex_run_planner.containers import RouteConfig, ChainMetrics
from rex_run_planner.data_prep.lidar import get_elevation
from rex_run_planner.data_prep.graph_utils import GraphUtils

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

    def load_graph(self, source_path: str) -> nx.Graph:
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

    def _remove_isolates(self):
        """Remove any nodes from the graph which are not connected to another
        node."""
        isolates = set(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolates)

    def _remove_dead_ends(self):
        """Remove any nodes which have been flagged for removal on account of
        them representing a dead end."""
        self.graph.remove_nodes_from(self.nodes_to_remove)

    def _refresh_node_lists(self):
        """Check the graph for any nodes which can be condensed, or removed
        entirely."""
        self.nodes_to_condense = []
        self.nodes_to_remove = []

        for node_id in self.graph.nodes:
            edges = self.graph.edges(node_id)
            node_degree = len(edges)

            if node_degree >= 3:
                # Node is a junction, must be retained
                continue
            elif node_degree == 2:
                # Node represents a bend in a straight line, can safely be
                # condensed
                self.nodes_to_condense.append(node_id)
            elif node_degree == 1:
                # Node is a dead end, can safely be removed
                self.nodes_to_remove.append(node_id)
            # Node is an orphan, will be caught by remove_isolates
            continue

    def _generate_node_chain(
        self, node_id: int
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """For a node with order 2, generate a chain which represents the
        traversal of both edges. For example if node_id is B and node B is
        connected to nodes A and C, the resultant chain would be [A, B, C].

        Args:
            node_id (int): The node which forms the middle of the chain

        Returns:
            Tuple[List[int], List[Tuple[int, int]]]: A list of the node IDs
              which are crossed as part of this chain, and a list of the
              edges which are traversed as part of this journey.
        """
        node_edges = list(self.graph.edges(node_id))
        node_chain = [node_edges[0][1], node_edges[0][0], node_edges[1][1]]

        return node_chain, node_edges

    def _calculate_chain_metrics(
        self, chain: List[int]
    ) -> Union[ChainMetrics, None]:
        """For a chain of 3 nodes, sum up the metrics for each of the edges
        which it is comprised of.

        Args:
            chain (List[int]): A list of 3 node IDs

        Returns:
            ChainMetrics: A container for the calculated metrics
        """
        try:
            # Fetch the two edges for the chain
            edge_1 = self.graph[chain[0]][chain[1]]
            edge_2 = self.graph[chain[1]][chain[2]]
        except KeyError:
            # Edge does not exist in this direction
            return None

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
        metrics = ChainMetrics(
            start=chain[0],
            end=chain[-1],
            gain=gain_1 + gain_2,
            loss=loss_1 + loss_2,
            dist=dist_1 + dist_2,
            vias=via_1 + [chain[1]] + via_2,
        )

        return metrics

    def _add_edge_from_chain_metrics(self, metrics: Union[ChainMetrics, None]):
        """Once metrics have been calculated for a node chain, use them to
        create a new edge which skips over the middle node. The ID of this
        middle node will be recorded within the `via` attribute of the new
        edge.

        Args:
            metrics (Union[ChainMetrics, None]): Container for calculated
              metrics for this chain
        """
        if metrics:
            self.graph.add_edge(
                metrics.start,
                metrics.end,
                via=metrics.vias,
                elevation_gain=metrics.gain,
                elevation_loss=metrics.loss,
                distance=metrics.dist,
            )

    def _remove_original_edges(self, node_edges: List[Tuple[int, int]]):
        """Once a new edge has been created based on a node chain, the
        original edges can be removed.

        Args:
            node_edges (List[Tuple[int, int]]): A list of node edges to be
              removed from the graph.
        """
        # Remove original edges
        for start, end in node_edges:
            try:
                self.graph.remove_edge(start, end)
            except NetworkXError:
                pass
            try:
                self.graph.remove_edge(end, start)
            except NetworkXError:
                pass

    def _condense_graph(self, _iter: int = 0):
        """Disconnect all nodes from the graph which contribute only
        geometrical information (i.e. they form corners along paths/roads but
        do not represent junctions). Update the edges in the graph to skip over
        these nodes, instead going direct from one junction to the next.

        Args:
            _iter (int, optional): The number of times this function has been
              called. Defaults to False.
        """
        self._remove_isolates()
        self._refresh_node_lists()

        # Early stopping condition
        if not self.nodes_to_condense:
            return

        iters = 0
        pbar = tqdm(
            desc=f"Condensing Graph (pass {_iter})",
            total=len(self.nodes_to_condense),
        )
        while self.nodes_to_condense:
            node_id = self.nodes_to_condense.pop()

            node_chain, node_edges = self._generate_node_chain(node_id)

            se_metrics = self._calculate_chain_metrics(node_chain)
            es_metrics = self._calculate_chain_metrics(node_chain[::-1])

            self._add_edge_from_chain_metrics(se_metrics)
            self._add_edge_from_chain_metrics(es_metrics)

            self._remove_original_edges(node_edges)

            self._refresh_node_lists()

            iters += 1
            pbar.total = len(self.nodes_to_condense) + iters
            pbar.update(1)
        pbar.close()

        self._remove_dead_ends()
        tqdm.write(f"Removed {len(self.nodes_to_remove)} dead ends")

        if _iter < self.config.max_condense_passes:
            self._condense_graph(_iter=_iter + 1)

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
            self._remove_isolates()
            self.save_graph(full_target_loc)

        self._condense_graph()

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
