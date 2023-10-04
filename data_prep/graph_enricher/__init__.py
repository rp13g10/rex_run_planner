"""Contains the GraphEnricher class, used to add additional data points
to the provided OSM graph."""

import json
import pickle
from typing import Optional

from tqdm import tqdm
from networkx import Graph
from networkx.readwrite import json_graph

from rex_run_planner.containers import RouteConfig
from rex_run_planner.data_prep.lidar import get_elevation
from rex_run_planner.data_prep.graph_utils import GraphUtils
from rex_run_planner.data_prep.graph_enricher.condenser import condense_graph
from rex_run_planner.data_prep.graph_enricher.tagger import tag_graph

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

        self.graph = tag_graph(self.graph, self.config)

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
