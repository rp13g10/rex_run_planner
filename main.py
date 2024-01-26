"""In the absence of a web frontend, this script is used to trigger library
calls and generate/plot routes"""

import os
import pickle


# pylint: disable=wrong-import-position
# ruff: noqa: E402
from rex_run_planner.containers import RouteConfig
from rex_run_planner.route_finding import RouteFinder
from rex_run_planner.route_plotting import (
    plot_route,
    generate_filename,
    RouteSelector,
)

# TODO: Parameterise 10% variation in max distance, remove all hard-coded
#       values
# TODO: Set up quick experiment to determine optimal no. candidates
# TODO: Build out unit tests for all code created so far
# TODO: Start building this out into a webapp once plots working properly
# TODO: Check handling of max_condense_passes, should be possible to remove

config = RouteConfig(
    # start_lat=50.969540,
    # start_lon=-1.383318,
    # Trying somewhere a bit further afield
    start_lat=51.049676,
    start_lon=-1.311284,
    max_distance=42,
    route_mode="flat",
    max_candidates=2048
)

DATA_DIR = '/home/ross/repos/refinement/data'

with open(os.path.join(DATA_DIR, 'condensed_graph.nx'), 'rb') as fobj:
    cond_graph = pickle.load(fobj)

finder = RouteFinder(graph=cond_graph, config=config)
routes = finder.find_routes()

# with open(os.path.join(DATA_DIR, "cached_routes.pkl"), 'wb') as fobj:
#     pickle.dump(routes, fobj)

# with open(os.path.join(DATA_DIR, "cached_routes.pkl"), 'rb') as fobj:
#     routes = pickle.load(fobj)

with open(os.path.join(DATA_DIR, 'full_graph.nx'), 'rb') as fobj:
    full_graph = pickle.load(fobj)

selector = RouteSelector(routes, num_routes_to_select=25, threshold=0.9)
selected_routes = selector.select_routes()


for route in selected_routes[:10]:
    plot = plot_route(full_graph, route)
    fname = generate_filename(route)
    plot.write_html(f"./plots/{config.route_mode}/{fname}.html")
