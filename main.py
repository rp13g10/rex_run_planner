"""In the absence of a web frontend, this script is used to trigger library
calls and generate/plot routes"""

# TODO: Fix this properly
import sys

sys.path.append("/mnt/c/rpiper/repos")

# pylint: disable=wrong-import-position
# ruff: noqa: E402
import pickle
from rex_run_planner.data_prep.graph_enricher import GraphEnricher
from rex_run_planner.route_finding import RouteFinder
from rex_run_planner.containers import RouteConfig
from rex_run_planner.route_plotting import (
    plot_route,
    generate_filename,
    RouteSelector,
)

"""
Plan:
 Phase 1 - Get something tangible
 * Set up route_finder as a separate package, to accept a coarsely filtered
   networkx graph as an input.
 * Fully paramaterize all aspects of the route finding algorithm, including
   edge types now that they're available in the graph
 * Build out a basic webapp as a PoC
 
 Phase 2 - Foundations for scalable webapp
 * Look in to options for improving cassandra cluster performance locally,
   suspect memory causing bottleneck at present
 * If feasible, switch refinement over to using graphframes rather than
   networkx for processing.
   
   https://docs.databricks.com/en/_extras/notebooks/source/graphframes-user-guide-py.html
   https://github.com/graphframes/graphframes/issues/408 - python package install
   jar file will also need adding as a python dependency
 
 Phase 3 - Set up the data preparation pipeline
  Single k8s stack with all required components
  * Cassandra database
  * pySpark cluster
  * airflow
    * triggers ingestion & processing when new files land
  * custom ingestion script
    https://docs.datastax.com/en/dsbulk/docs/overview/dsbulk-about.html
    Loads data from lidar files into the database
  * custom tagging script
    * triggers tagging of nodes/edges with elevation data once ingestion completes
  * custom parsing script
    osm to networkx, one-off run, arguably could be left out of stack

 Phase 4 - Full build
  * Set up a separate k8s stack for the webapp (or add to existing? check
    suggested design patterns)
  * Move things over to the cloud
"""

# TODO: Parameterise 10% variation in max distance, remove all hard-coded
#       values
# TODO: Set up quick experiment to determine optimal no. candidates
# TODO: Build out unit tests for all code created so far
# TODO: Start building this out into a webapp once plots working properly
# TODO: Check handling of max_condense_passes, should be possible to remove

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

del graph
with open("./data/hampshire-latest-full.nx", "rb") as fobj:
    graph = pickle.load(fobj)

selector = RouteSelector(routes, num_routes_to_select=25, threshold=0.85)
selected_routes = selector.select_routes()


for route in selected_routes[:10]:
    plot = plot_route(graph, route)
    fname = generate_filename(route)
    plot.write_html(f"./plots/{fname}.html")
