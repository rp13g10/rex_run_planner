"""In the absence of a web frontend, this script is used to trigger library
calls and generate/plot routes"""

# TODO: Fix this properly
import sys

sys.path.append("/mnt/c/rpiper/repos")

# ruff: noqa: E402
import pickle
from rex_run_planner.data_prep import GraphEnricher
from rex_run_planner.route_finding import RouteFinder
from rex_run_planner.containers import RouteConfig

# TODO: Build out unit tests for all code created so far
# TODO: Start building this out into a webapp once plots working properly

config = RouteConfig(
    start_lat=50.969540,
    start_lon=-1.383318,
    max_distance=10,
    route_mode="hilly",
    dist_mode="metric",
    elevation_interval=10,
    max_candidates=32000,
    max_condense_passes=5,
)

try:
    with open("./data/hampshire-latest-cond.nx", "rb") as fobj:
        graph = pickle.load(fobj)
except FileNotFoundError:
    enricher = GraphEnricher("./data/hampshire-latest.json", config)
    enricher.enrich_graph(
        full_target_loc="./data/hampshire-latest-full.nx",
        cond_target_loc="./data/hampshire-latest-cond.nx",
    )
    graph = enricher.graph
    del enricher

finder = RouteFinder(graph=graph, config=config)
routes = finder.find_routes()