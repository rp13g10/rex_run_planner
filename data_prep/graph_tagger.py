from networkx import Graph
from graph_utils import GraphUtils

class GraphTagger(GraphUtils):

    def __init__(
        self,
        graph: Graph,
        dist_mode: str = "metric",
        elevation_interval: int = 10,
    ):
        """Create an instance of the graph tagger class with a user-provided
        graph.

        Args:
            graph (Graph): The networkx graph to be tagged
            dist_mode (str, optional): The preferred output mode for distances
              which are saved to node edges. Returns kilometers if set to
              metric, miles if set to imperial. Defaults to "metric".
            elevation_interval (int, optional): When calculating elevation
              changes across an edge, values will be estimated by taking
              checkpoints at regular checkpoints. Smaller values will result in
              more accurate elevation data, but may slow down the calculation.
              Defaults to 10.
        """

        super().__init__(graph, dist_mode, elevation_interval)



    def tag_distances_to_start(self, start_node: int):
        """Tags each node in the graph with the distance & elevation which
        must be travelled (in a straight line) in order to get back to the
        start point."""

        for node_id in self.graph.nodes:
            # TODO: Create simpler function to get delta between two nodes
            dist_to_start, gain_to_start, loss_to_start = self._estimate_distance_and_elevation_change(node_id, start_node)

            self.graph.nodes[node_id]["dist_to_start"] = dist_to_start
            self.graph.nodes[node_id]["gain_to_start"] = gain_to_start
            self.graph.nodes[node_id]["loss_to_start"] = loss_to_start

    # TODO: Add in subgraph creation:
    #       Use simple bbox
    #       Tag distances to start
    #       Use calculated distances