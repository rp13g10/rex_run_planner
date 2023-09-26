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
from rex_run_planner.route_plotting import (
    plot_route,
    generate_filename,
    RouteSelector,
)

# TODO: Parameterise 10% variation in max distance, remove all hard-coded
#       values

# TODO: Build out unit tests for all code created so far
# TODO: Start building this out into a webapp once plots working properly

config = RouteConfig(
    start_lat=50.969540,
    start_lon=-1.383318,
    max_distance=10,
    route_mode="hilly",
    dist_mode="metric",
    elevation_interval=10,
    max_candidates=16000,
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
cands = finder.last_candidates

del graph
with open("./data/hampshire-latest-full.nx", "rb") as fobj:
    graph = pickle.load(fobj)

for route in cands[:10]:
    plot = plot_route(graph, route)
    fname = generate_filename(route)
    plot.write_html(f"./plots/cand_{fname}.html")

selector = RouteSelector(routes, num_routes_to_select=25, threshold=0.75)
selected_routes = selector.select_routes()


for route in selected_routes[:10]:
    plot = plot_route(graph, route)
    fname = generate_filename(route)
    plot.write_html(f"./plots/{fname}.html")
