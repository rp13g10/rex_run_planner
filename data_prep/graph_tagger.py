"""Defines the GraphTagger class, which tags each node in the graph with its
distance from the start point"""

from networkx import Graph, shortest_path_length
from networkx.exception import NetworkXNoPath
from rex_run_planner.containers import RouteConfig, BBox
from rex_run_planner.data_prep.graph_utils import GraphUtils

# TODO: Change distance to start calculations to use shortest path rather than
#       straight line distance, may save some wasted compute


class GraphTagger(GraphUtils):
    """Before creating a set of new routes, use the GraphTagger class to
    pre-calculate the distance of each node to the start point. This
    information can then be used to reduce the size of the graph, minimising
    its memory footprint."""

    def __init__(
        self,
        graph: Graph,
        config: RouteConfig,
    ):
        """Create an instance of the graph tagger class with a user-provided
        graph.

        Args:
            graph (Graph): The networkx graph to be tagged
            dist_mode (str, optional): The preferred output mode for distances
              which are saved to node edges. Returns kilometers if set to
              metric, miles if set to imperial. Defaults to "metric".
            elevation_interval (Union[float, int], optional): When calculating
              elevation changes across an edge, values will be estimated by
              taking samples at regular checkpoints. Smaller values will result
              in more accurate elevation data, but may slow down the
              calculation. Defaults to 10.
        """

        super().__init__(graph, config)

    def _check_if_node_is_in_target_area(
        self, node_id: int, bbox: BBox
    ) -> bool:
        """Check whether the a node falls within the provided bounding box.

        Args:
            node_id (int): The node to be checked
            bbox (BBox): The bounding box to be checked

        Returns:
            bool: True if node_id is inside bbox, else False
        """

        node_lat = self.graph.nodes[node_id]["lat"]
        node_lon = self.graph.nodes[node_id]["lon"]

        lat_check = bbox.min_lat <= node_lat <= bbox.max_lat
        if not lat_check:
            return False

        lon_check = bbox.min_lon <= node_lon <= bbox.max_lon
        return lon_check

    def generate_coarse_subgraph(self, start_node: int) -> Graph:
        """Given a starting node and a max distance, roughly trim the
        internal graph to a grid square of edge size ~ max distance.

        Args:
            start_node (int): The starting point of the route

        Returns:
            Graph: A subset of the internal graph
        """
        start_lat = self.graph.nodes[start_node]["lat"]
        start_lon = self.graph.nodes[start_node]["lon"]

        max_dist = self.config.max_distance
        dist_mode = self.config.dist_mode

        # NOTE: 1 degree ~ 69mi/111km
        factor = 111 if dist_mode == "metric" else 69

        # 5% safety net as this is a very crude estimate
        delta = (max_dist / 2) / (factor * 1.05)

        # TODO: Get a refined estimate for lat/lon per mile/km, which respects
        #       the specified distance mode
        bbox = BBox(
            min_lat=start_lat - delta,
            max_lat=start_lat + delta,
            min_lon=start_lon - delta,
            max_lon=start_lon + delta,
        )

        nodes_to_remove = list(
            filter(
                lambda node: not self._check_if_node_is_in_target_area(
                    node, bbox
                ),
                self.graph.nodes,
            )
        )

        self.graph.remove_nodes_from(nodes_to_remove)
        return self.graph

    def tag_distances_to_start(self, start_node: int) -> Graph:
        """Tags each node in the graph with the distance & elevation which
        must be travelled (in a straight line) in order to get back to the
        start point."""

        assert (
            start_node in self.graph.nodes
        ), "Start node {start_node} is not in the graph!"

        for node_id in self.graph.nodes:
            try:
                dist_to_start = shortest_path_length(
                    self.graph, node_id, start_node, weight="distance"
                )
            except NetworkXNoPath:
                dist_to_start = None

            (
                _,
                gain_to_start,
                loss_to_start,
            ) = self._get_straight_line_distance_and_elevation_change(
                start_node, node_id
            )

            self.graph.nodes[node_id]["dist_to_start"] = dist_to_start
            self.graph.nodes[node_id]["gain_to_start"] = gain_to_start
            self.graph.nodes[node_id]["loss_to_start"] = loss_to_start

        return self.graph

    def generate_fine_subgraph(self) -> Graph:
        """After tagging each node with its distance from the start point, use
        this information to remove all nodes which cannot be reached without
        the route going over the maximum configured distance.

        Returns:
            Graph: A fully trimmed graph where all nodes are a useful distance
              from the start point.
        """
        nodes_to_remove = set()
        for node_id in self.graph.nodes:
            node_dist = self.graph.nodes[node_id]["dist_to_start"]
            if node_dist is None:
                nodes_to_remove.add(node_id)
            elif node_dist > (self.config.max_distance * 1.1) / 2:
                nodes_to_remove.add(node_id)

        self.graph.remove_nodes_from(nodes_to_remove)
        return self.graph
