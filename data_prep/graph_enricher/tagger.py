"""Contains the GraphTagger class, which will be executed if this script
is executed directly."""

from itertools import repeat

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from networkx import Graph

from relevation.lidar import get_elevation

from rex_run_planner.containers import RouteConfig

# from rex_run_planner.data_prep.lidar import get_elevation
from rex_run_planner.data_prep.graph_utils import GraphUtils
from rex_run_planner.data_prep.graph_enricher.splitter import GraphSplitter


class GraphTagger(GraphUtils):
    """Class which enriches the data which is provided by Open Street Maps.
    Unused data is stripped out, and elevation data is added for both nodes and
    edges. The graph itself is condensed, with nodes that lead to dead ends
    or only represent a bend in the route being removed.
    """

    def __init__(
        self,
        graph: Graph,
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

        # Store down core attributes
        super().__init__(graph, config)

    def _enrich_source_nodes(self):
        """For each node in the graph, attempt to fetch elevation info from
        the loaded LIDAR data. If no elevation information is available, the
        node will be dropped to minimise memory usage."""

        sorted_nodes = sorted(
            sorted(
                self.graph.nodes.items(),
                key=lambda item: item[1]["lat"],
            ),
            key=lambda item: item[1]["lon"],
        )

        to_delete = set()
        for node_id, node_attrs in sorted_nodes:
            # Unpack coordinates
            lat = node_attrs["lat"]
            lon = node_attrs["lon"]

            # Fetch elevation
            try:
                elevation = get_elevation(lat, lon)
            except FileNotFoundError:
                elevation = None

            if not elevation:
                # Mark node for deletion
                to_delete.add(node_id)
            else:
                # Add elevation to node
                self.graph.nodes[node_id]["elevation"] = elevation

        # Remove nodes with no elevation data
        self.graph.remove_nodes_from(to_delete)

    def _enrich_source_edges(self):
        """For each edge in the graph, estimate the distance, elevation gain
        and elevation loss when traversing it. Strip out all other metadata
        to minimise the memory footprint of the graph.
        """

        sorted_edges = sorted(
            sorted(
                self.graph.edges(data=True),
                key=lambda edge: self.fetch_node_coords(edge[0])[0],
            ),
            key=lambda edge: self.fetch_node_coords(edge[0])[0],
        )

        # Calculate elevation change & distance for each edge
        for start_id, end_id, data in sorted_edges:
            if "distance" in data:
                continue

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

    def tag_graph(
        self,
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


def _tag_subgraph(subgraph: Graph, config: RouteConfig):
    tagger = GraphTagger(subgraph, config)
    tagger.tag_graph()

    return tagger.graph


def _tag_subgraph_star(args):
    return _tag_subgraph(*args)


def tag_graph(graph: Graph, config: RouteConfig):
    # Split the graph across a grid
    splitter = GraphSplitter(graph, no_subgraphs=1000)
    splitter.explode_graph()

    # Process each grid separately
    map_args = zip(splitter.subgraphs.values(), repeat(config))

    # TODO: Build this functionality into the splitter class
    new_subgraphs = process_map(
        _tag_subgraph_star,
        map_args,
        desc="Enriching subgraphs",
        tqdm_class=tqdm,
        total=len(splitter.grid),
        # max_workers=8,
    )

    # Re-combine the tagged subgraphs
    # populated_squares = set()
    for new_subgraph in new_subgraphs:
        subgraph_id = new_subgraph.graph["grid_square"]
        splitter.subgraphs[subgraph_id] = new_subgraph

    # Clear out any grid squares not preset in the output
    # for subgraph_id in list(splitter.subgraphs.keys()):
    #     if subgraph_id not in populated_squares:
    #         del splitter.subgraphs[subgraph_id]
    splitter.rebuild_graph()

    # Perform a mop-up to calculate elevation changes for the edges which
    # spanned more than one subgraph
    graph = _tag_subgraph(graph, config)

    return graph
