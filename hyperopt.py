"""In the absence of a web frontend, this script is used to trigger library
calls and generate/plot routes"""

# TODO: Fix this properly
import sys

sys.path.append("/mnt/c/rpiper/repos")

# ruff: noqa: E402

from datetime import datetime
import pickle
import pandas as pd
from rex_run_planner.data_prep import GraphEnricher
from rex_run_planner.route_finding import RouteFinder
from rex_run_planner.containers import RouteConfig


base_config = RouteConfig(
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
    enricher = GraphEnricher("./data/hampshire-latest.json", base_config)
    enricher.enrich_graph(
        full_target_loc="./data/hampshire-latest-full.nx",
        cond_target_loc="./data/hampshire-latest-cond.nx",
    )
    graph = enricher.graph
    del enricher

results = []
for cands in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:  # 32768]:
    start_time = datetime.now()
    config = RouteConfig(
        start_lat=50.969540,
        start_lon=-1.383318,
        max_distance=10,
        route_mode="hilly",
        dist_mode="metric",
        elevation_interval=10,
        max_candidates=cands,
        max_condense_passes=5,
    )

    finder = RouteFinder(graph=graph, config=config)
    routes = finder.find_routes()
    best_route = routes[0]
    best_elevation = best_route.elevation_gain
    n_routes = len(routes)
    end_time = datetime.now()

    exec_time = end_time - start_time
    exec_time = exec_time.seconds

    result = {
        "n_cands": cands,
        "exec_time": exec_time,
        "routes_found": n_routes,
        "best_elevation": best_elevation,
    }
    results.append(result)

df = pd.DataFrame.from_records(results)

df.to_csv("./experiment_results_hilly.csv", sep="\t")
