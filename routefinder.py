import difflib
import random

# from rapidfuzz import fuzz
import numpy as np
import redis
import tqdm

from copy import deepcopy
from plotly import graph_objects as go

from maputils import (
    find_nearest_node,
    calculate_distance_and_elevation,
    fetch_node_coords,
)
from enrich_networkx import osm


pool = redis.ConnectionPool(host="localhost", port=6379, db=0)
cache = redis.Redis(connection_pool=pool)

# TODO: For final output, use difflib to ensure each suggested route is
#       significantly different from the next.


def prune_routes(routes, max_routes, max_distance):

    if len(routes) < max_routes:
        return routes

    all_nodes = set()
    for route in routes:
        nodes = route["route"]
        nodes = set(nodes)
        all_nodes.update(nodes)

    all_lats = set()
    all_lons = set()
    for node in all_nodes:
        lat, lon = fetch_node_coords(osm, node)
        all_lats.add(lat)
        all_lons.add(lon)

    # Grid size up to 0.5*0.5km. Make this static in a future build
    # TODO: Filter out bins which don't contain any nodes? Possibly generate
    #       tuples (boundaries) to iterate over first.
    #       Any way to generate & divide up a shapefile?
    lat_bins = np.linspace(min(all_lats), max(all_lats), int(max_distance * 2))
    lon_bins = np.linspace(min(all_lons), max(all_lons), int(max_distance * 2))

    min_lat = min(all_lats)
    min_lon = min(all_lons)

    routes = sorted(routes, key=lambda x: x["ratio"], reverse=True)

    def get_matching_routes(routes, min_lat, max_lat, min_lon, max_lon):

        assert min_lat != max_lat
        assert min_lon != max_lon

        num_squares = ((max_distance * 2) - 1) ** 2

        to_keep = []
        for route in routes:
            last_pos = route["route"][-1]
            lat, lon = fetch_node_coords(osm, last_pos)
            if min_lat <= lat < max_lat and min_lon <= lon < max_lon:
                to_keep.append(route)
            if len(to_keep) == int(max_routes / num_squares):
                return to_keep
        return to_keep

    # TODO: If number of kept routes below max, pull through remaining with
    #       random samples. Or, collect all routes and trim to required
    #       length afterwards.

    kept_routes = []
    min_lat = None
    min_lon = None
    for lat_bin in tqdm.tqdm(lat_bins, desc="Pruning routes", leave=False):

        if min_lat is None:
            min_lat = lat_bin
            continue
        for lon_bin in lon_bins:

            if min_lon is None:
                min_lon = lon_bin
                continue

            square_routes = get_matching_routes(
                routes, min_lat, lat_bin, min_lon, lon_bin
            )
            kept_routes += square_routes  # type: ignore

            min_lon = lon_bin
        min_lat = lat_bin

    return kept_routes


def find_routes(
    lat, lon, max_distance, target_elevation="max", max_candidates=50000
):

    # TODO: Implement support for multi-threading, investigate use of
    #       shared memory for network graph

    # TODO: Check impact of max_candidates on quality of returned routes,
    #       how low can it be set while returing good quality results?

    start_node = find_nearest_node(osm, lat, lon)

    seed = {
        "route": [start_node],
        "visited": {start_node},
        "distance": 0,
        "elevation_gain": 0,
        "elevation_loss": 0,
        "id": "0_0",
    }

    valid_routes = []
    candidate_routes = [seed]

    pbar = tqdm.tqdm()
    iters = 0
    while candidate_routes:

        new_candidates = []
        for cand_inx, candidate in enumerate(candidate_routes):
            cur_pos = candidate["route"][-1]
            start_pos = candidate["route"][0]
            visited = candidate["visited"]
            neighbours = filter(
                lambda x: x not in visited, osm.neighbors(cur_pos)
            )

            for neigh_inx, neighbour in enumerate(neighbours):

                route_id = f"{cand_inx}_{neigh_inx}"

                new_candidate = deepcopy(candidate)
                new_candidate["route"].append(neighbour)
                new_candidate["visited"].add(neighbour)
                new_candidate["id"] = route_id

                # Allow routes to finish at the start node
                if iters == 2:
                    new_candidate["visited"].remove(start_pos)

                cur_edge = osm[cur_pos][neighbour]
                distance = cur_edge["distance"]
                elevation = cur_edge["elevation"]

                new_candidate["distance"] += distance
                if elevation >= 0:
                    new_candidate["elevation_gain"] += elevation
                else:
                    new_candidate["elevation_loss"] -= elevation

                new_candidate["ratio"] = (
                    new_candidate["elevation_gain"] / new_candidate["distance"]
                )

                # Stop if route can't get back to start position without going
                # over max_distance
                if new_candidate["distance"] > max_distance / 2:
                    cache_key = f"{start_pos}|{neighbour}"
                    dist_to_start = cache.get(cache_key)
                    if dist_to_start:
                        dist_to_start = float(dist_to_start)
                    else:
                        dist_to_start, _ = calculate_distance_and_elevation(
                            osm, start_pos, neighbour
                        )
                        cache.set(cache_key, dist_to_start)

                    dist_remaining = max_distance - new_candidate["distance"]
                    if dist_to_start > dist_remaining:
                        continue

                if new_candidate["distance"] < max_distance * 1.05:

                    if neighbour == start_pos:
                        if new_candidate["distance"] >= max_distance * 0.95:
                            # TODO: Strip out redundant datapoints before storing
                            valid_routes.append(new_candidate)
                    else:
                        new_candidates.append(new_candidate)

        # if not iters % 1:
        #     new_candidates = prune_routes(new_candidates)

        candidate_routes = new_candidates
        n_candidates = len(candidate_routes)

        if n_candidates > max_candidates:
            # Proposal - Move this into prune_routes, sort by lat & long to
            # ensure retained routes are evenly spread across the local area
            # Alternatively, check proximity to local maxima and discard flat
            # routes which are far away from any hills
            candidate_routes = prune_routes(
                candidate_routes, max_candidates, max_distance
            )
            # candidate_routes = random.sample(candidate_routes, max_candidates)
            # candidate_routes = sorted(
            #     candidate_routes,
            #     key=lambda x: x["elevation_gain"],
            #     reverse=target_elevation == "max",
            # )
            # candidate_routes = candidate_routes[:max_candidates]
            n_candidates = len(candidate_routes)
            # TODO - Recalculate iter_dist using only retained reoutes

        if n_candidates:
            iter_dist = sum(x["distance"] for x in candidate_routes)
            avg_distance = iter_dist / n_candidates
        else:
            avg_distance = max_distance

        n_valid = len(valid_routes)
        iters += 1
        pbar.update(1)
        pbar.set_description(
            (
                f"{n_candidates:,.0f} cands | {n_valid:,.0f} valid | "
                f"{avg_distance:,.2f} avg dist"
            )
        )

    pbar.close()

    valid_routes = sorted(
        valid_routes,
        key=lambda x: x["elevation_gain"],
        reverse=target_elevation == "max",
    )

    return valid_routes


def plot_route(route, mode="metric"):
    lats, lons = [], []
    for node_id in route["route"]:
        lat, lon = fetch_node_coords(osm, node_id)
        lats.append(lat)
        lons.append(lon)

    trace = go.Scattermapbox(mode="lines", lat=lats, lon=lons)

    distance = route["distance"]
    elevation = route["elevation_gain"]
    title = f"Distance: {distance}, Elevation: {elevation}"

    layout = go.Layout(
        # margin={"l": 0, "t": 0, "r": 0, "l": 0},
        mapbox={"center": {"lon": lons[0], "lat": lats[0]}},
        mapbox_style="stamen-terrain",
        mapbox_zoom=10,
        title=title,
    )

    fig = go.Figure(data=[trace], layout=layout)

    return fig


target_distance = 17.0
valid_routes = find_routes(
    50.969540, -1.383318, target_distance, max_candidates=32000
)

for inx, route in enumerate(valid_routes[:50]):
    fig = plot_route(route)
    dist = route["distance"]
    elev = route["elevation_gain"]
    fig.write_html(f"plots/hilly/{target_distance}_{inx}_{elev}.html")

for inx, route in enumerate(valid_routes[-50:]):
    fig = plot_route(route)
    dist = route["distance"]
    elev = route["elevation_gain"]
    fig.write_html(f"plots/flat/{target_distance}_{inx}_{elev}.html")

import pickle

with open("data/all_routes.pkl", "wb") as fobj:
    pickle.dump(valid_routes, fobj)

# docker run -p 6379:6379 -it redis/redis-stack:latest
