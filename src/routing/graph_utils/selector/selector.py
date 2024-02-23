from typing import Tuple

from geopy import distance, point
from graphframes import GraphFrame
from networkx import DiGraph
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

from routing.containers.routes import RouteConfig
from routing.containers.pruning import BBox
from routing.route_maker.utilities import find_nearest_node


class Selector:

    def __init__(
        self,
        config: RouteConfig,
    ):
        """Create an instance of the graph enricher class based on the
        contents of the networkx graph specified by `source_path`

        Args:
            source_path (str): The location of the networkx graph to be
              enriched. The graph must have been saved to json format.
            config (RouteConfig): A configuration file detailing the route
              requested by the user.
        """

        # Store down core attributes
        self.config = config

        # Generate internal sparkcontext
        conf = SparkConf()
        conf = conf.setAppName("refinement")
        conf = conf.setMaster("local[10]")
        conf = conf.set("spark.driver.memory", "2g")
        conf = conf.set(
            "spark.jars.packages",
            "graphframes:graphframes:0.8.3-spark3.5-s_2.12",
        )
        conf = conf.set("spark.jars", "graphframes-0.8.3-spark3.5-s_2.12.jar")

        sc = SparkContext(conf=conf)
        sc.setLogLevel("WARN")
        self.sc = sc.getOrCreate()
        self.sql = SQLContext(self.sc)

        self.graph = self.create_graph()

    def create_graph(self) -> GraphFrame:

        nodes = self.sql.read.parquet(
            "/home/ross/repos/refinement/data/enriched_nodes"
        )

        edges = self.sql.read.parquet(
            "/home/ross/repos/refinement/data/enriched_edges"
        )

        graph = GraphFrame(nodes, edges)

        return graph

    def get_bounding_box_for_route(self) -> BBox:

        start_point = point.Point(self.config.start_lat, self.config.start_lon)

        dist_to_corner = (self.config.max_distance / 2) * (2**0.5)

        nw_corner = distance.distance(kilometers=dist_to_corner).destination(
            point=start_point, bearing=315
        )

        se_corner = distance.distance(kilometers=dist_to_corner).destination(
            point=start_point, bearing=135
        )

        bbox = BBox(
            min_lat=se_corner.latitude,
            min_lon=nw_corner.longitude,
            max_lat=nw_corner.latitude,
            max_lon=se_corner.longitude,
        )

        return bbox

    def convert_graphframe_to_nx(self, graph: GraphFrame) -> DiGraph:

        nodes = graph.vertices.collect()
        edges = graph.edges.collect()

        nodes = [
            (
                row.id,
                {"lat": row.lat, "lon": row.lon, "elevation": row.elevation},
            )
            for row in nodes
        ]

        edges = [
            (
                row.src,
                row.dst,
                {
                    "distance": row.distance,
                    "elevation_gain": row.elevation_gain,
                    "elevation_loss": row.elevation_loss,
                    "type": row.type,
                },
            )
            for row in edges
        ]

        nx_graph = DiGraph()
        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)

        return nx_graph

    def fetch_node_coords(
        self, graph: DiGraph, node_id: int
    ) -> Tuple[int, int]:
        """Convenience function, retrieves the latitude and longitude for a
        single node in a graph."""
        node = graph.nodes[node_id]
        lat = node["lat"]
        lon = node["lon"]
        return lat, lon

    def _get_straight_line_distance_and_elevation_change(
        self, graph: DiGraph, start_id: int, end_id: int
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
        start_ele = graph.nodes[start_id]["elevation"]
        end_ele = graph.nodes[end_id]["elevation"]
        change = end_ele - start_ele
        gain = max(0, change)
        loss = abs(min(0, change))

        # Distance change
        start_lat, start_lon = self.fetch_node_coords(graph, start_id)
        end_lat, end_lon = self.fetch_node_coords(graph, end_id)
        dist = distance.distance((start_lat, start_lon), (end_lat, end_lon))
        dist = dist.kilometers

        return dist, gain, loss

    def tag_distances_to_start(
        self, graph: DiGraph, start_node: int
    ) -> DiGraph:
        """Tags each node in the graph with the distance & elevation which
        must be travelled (in a straight line) in order to get back to the
        start point."""

        assert (
            start_node in graph.nodes
        ), "Start node {start_node} is not in the graph!"

        for node_id in graph.nodes:

            (
                dist_to_start,
                gain_to_start,
                loss_to_start,
            ) = self._get_straight_line_distance_and_elevation_change(
                graph, start_node, node_id
            )

            graph.nodes[node_id]["dist_to_start"] = dist_to_start
            graph.nodes[node_id]["gain_to_start"] = gain_to_start
            graph.nodes[node_id]["loss_to_start"] = loss_to_start

        return graph

    def generate_fine_subgraph(self, graph: DiGraph) -> DiGraph:
        """After tagging each node with its distance from the start point, use
        this information to remove all nodes which cannot be reached without
        the route going over the maximum configured distance.

        Returns:
            Graph: A fully trimmed graph where all nodes are a useful distance
              from the start point.
        """
        nodes_to_remove = set()
        for node_id in graph.nodes:
            node_dist = graph.nodes[node_id]["dist_to_start"]
            if node_dist is None:
                nodes_to_remove.add(node_id)
            elif node_dist > (self.config.max_distance) / 2:
                nodes_to_remove.add(node_id)

        graph.remove_nodes_from(nodes_to_remove)
        return graph

    def retrieve_networkx_graph(self) -> DiGraph:

        bbox = self.get_bounding_box_for_route()

        subgraph = (
            self.graph.filterVertices(f"lat >= {bbox.min_lat}")
            .filterVertices(f"lat <= {bbox.max_lat}")
            .filterVertices(f"lon >= {bbox.min_lon}")
            .filterVertices(f"lon <= {bbox.max_lon}")
        )

        if self.config.terrain_types:
            type_list = ", ".join(
                f"'{type}'" for type in self.config.terrain_types
            )
            subgraph = subgraph.filterEdges(f"type IN ({type_list})")

        subgraph = subgraph.dropIsolatedVertices()

        nx_graph = self.convert_graphframe_to_nx(subgraph)

        start_id = find_nearest_node(
            nx_graph, self.config.start_lat, self.config.start_lon
        )

        nx_graph = self.tag_distances_to_start(nx_graph, start_id)
        nx_graph = self.generate_fine_subgraph(nx_graph)

        return nx_graph
