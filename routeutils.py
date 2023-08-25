import numpy as np
import redis
import tqdm

from copy import deepcopy
from plotly import graph_objects as go

from maputils import (
    fetch_node_coords,
    calculate_distance_and_elevation,
    find_nearest_node,
)

pool = redis.ConnectionPool(host="localhost", port=6379, db=0)
cache = redis.Redis(connection_pool=pool)


def prune_routes(graph, routes, max_routes, max_distance, target_elevation):
    """
    For a set of candidate routes, retain only the most promising routes in
    terms of elevation gain wrt. the criteria set (max/min elevation).
    The area containing the routes will be divided into a grid, and an equal
    number of routes selected from each square. This ensures that areas with
    a higher density of paths don't end up dominating the algorithm.
    """
    # Only prune routes when required
    if len(routes) < max_routes:
        return routes

    # Get all of the nodes visited by any of the candidate routes
    all_nodes = set()
    for route in routes:
        nodes = route["route"]
        nodes = set(nodes)
        all_nodes.update(nodes)

    # Fetch the latitudes/longitudes for these routes
    all_lats = set()
    all_lons = set()
    for node in all_nodes:
        lat, lon = fetch_node_coords(graph, node)
        all_lats.add(lat)
        all_lons.add(lon)

    # Create a grid which covers the bounding box for all of the nodes which
    # have been visisted by a candidate route

    # NOTE: max_distance is used to set the number of squares in the grid, as
    #       it is assumed longer routes will cover a larger area

    # TODO: Filter out bins which don't contain any nodes? Possibly generate
    #       tuples (boundaries) to iterate over first.
    #       Any way to generate & divide up a shapefile?
    lat_bins = np.linspace(
        min(all_lats), max(all_lats), int(max_distance * 2)  # type: ignore
    )
    lon_bins = np.linspace(
        min(all_lons), max(all_lons), int(max_distance * 2)  # type: ignore
    )

    # min_lat = min(all_lats)
    # min_lon = min(all_lons)

    # Sort all routes according to the gradient across the whole route
    routes = sorted(
        routes,
        key=lambda x: x["ratio"],
        reverse=target_elevation == "max",
    )

    def get_matching_routes(routes, min_lat, max_lat, min_lon, max_lon):
        # Check that the square being searched is not a single point
        assert min_lat != max_lat
        assert min_lon != max_lon

        # Get the number of squares in the grid
        num_squares = ((max_distance * 2) - 1) ** 2

        # Find all routes which end within the current grid square
        to_keep = []
        for route in routes:
            last_pos = route["route"][-1]
            lat, lon = fetch_node_coords(graph, last_pos)
            # Retain the most promising routes
            if min_lat <= lat < max_lat and min_lon <= lon < max_lon:
                to_keep.append(route)
            # Once max routes for this grid square have been reached, drop
            # all remaining routes in favour of the more promising candidates
            if len(to_keep) == int(max_routes / num_squares):
                return to_keep
        return to_keep

    # TODO: If number of kept routes below max, pull through remaining with
    #       random samples. Or, collect all routes and trim to required
    #       length afterwards.

    # Prune routes, evenly distributing candidates across the search area
    kept_routes = []
    min_lat = None
    min_lon = None
    for lat_bin in tqdm.tqdm(lat_bins, desc="Pruning routes", leave=False):
        # Skip first iteration as only 1 boundary provided
        if min_lat is None:
            min_lat = lat_bin
            continue
        for lon_bin in lon_bins:
            # Skip first iteration as only 1 boundary provided
            if min_lon is None:
                min_lon = lon_bin
                continue

            # Get most promising routes for each square
            square_routes = get_matching_routes(
                routes, min_lat, lat_bin, min_lon, lon_bin
            )
            kept_routes += square_routes  # type: ignore

            min_lon = lon_bin
        min_lat = lat_bin

    return kept_routes


def find_routes(
    graph, lat, lon, max_distance, target_elevation, max_candidates=50000
):
    # TODO: Implement support for multi-threading, investigate use of
    #       shared memory for network graph

    # TODO: Check impact of max_candidates on quality of returned routes,
    #       how low can it be set while returing good quality results?

    # Find the starting node
    start_node = find_nearest_node(graph, lat, lon)

    # Create the first route, of 0 length
    seed = {
        "route": [start_node],
        "visited": {start_node},
        "distance": 0,
        "elevation_gain": 0,
        "elevation_loss": 0,
        "id": "0_0",
    }

    # Initialize containers
    valid_routes = []
    final_candidates = []
    candidate_routes = [seed]

    pbar = tqdm.tqdm()
    iters = 0
    while candidate_routes:
        new_candidates = []
        for cand_inx, candidate in enumerate(candidate_routes):
            # Get all nodes which can be reached from the current end point
            # of the route
            cur_pos = candidate["route"][-1]
            start_pos = candidate["route"][0]
            visited = candidate["visited"]

            # Get rid of any which have already been visited
            neighbours = filter(
                lambda x: x not in visited, graph.neighbors(cur_pos)
            )

            # Create a new candidate route for each neighbour which has not
            # been visited yet
            for neigh_inx, neighbour in enumerate(neighbours):
                # Generate an ID for each route
                route_id = f"{cand_inx}_{neigh_inx}"

                # Get stats for the latest step taken
                cur_edge = graph[cur_pos][neighbour]
                distance = cur_edge["distance"]
                elevation_gain = cur_edge["elevation_gain"]
                elevation_loss = cur_edge["elevation_loss"]
                # TODO - Switch to standard get once enrich_networkx reruns
                #        with fix in place
                vias = cur_edge.get("via", [])

                # Add the neighbour as the new end point of the route, along
                # with any intermediate nodes which are passed along the way
                new_candidate = deepcopy(candidate)
                new_candidate["route"] += vias
                new_candidate["route"].append(neighbour)
                new_candidate["visited"].add(neighbour)
                new_candidate["id"] = route_id

                # Allow routes to finish at the start node
                if iters == 2:
                    new_candidate["visited"].remove(start_pos)

                # Add metrics to the route total
                new_candidate["distance"] += distance
                new_candidate["elevation_gain"] += elevation_gain
                new_candidate["elevation_loss"] += elevation_loss

                # Calculate ratio/gradient across the route
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
                        dist_to_start, _, _ = calculate_distance_and_elevation(
                            graph, start_pos, neighbour
                        )
                        cache.set(cache_key, dist_to_start)

                    dist_remaining = max_distance - new_candidate["distance"]
                    if dist_to_start > dist_remaining:
                        continue

                # Save the route as either a candidate, or a completed circuit
                # TODO: Set 1.1/0.9 up as a parameter
                if new_candidate["distance"] < max_distance * 1.1:
                    if neighbour == start_pos:
                        if new_candidate["distance"] >= max_distance * 0.9:
                            valid_routes.append(new_candidate)
                    else:
                        new_candidates.append(new_candidate)

        # Update the list of candidate routes with the new one
        final_candidates = candidate_routes
        candidate_routes = new_candidates
        n_candidates = len(candidate_routes)

        # Prune the list of candidates if necessary
        if n_candidates > max_candidates:
            candidate_routes = prune_routes(
                graph,
                candidate_routes,
                max_candidates,
                max_distance,
                target_elevation,
            )

            n_candidates = len(candidate_routes)

        # Calculate the average length of the candidate routes
        if n_candidates:
            iter_dist = sum(x["distance"] for x in candidate_routes)
            avg_distance = iter_dist / n_candidates
        else:
            avg_distance = max_distance

        # Update the progress bar
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

    # Sort the routes before returning
    valid_routes = sorted(
        valid_routes,
        key=lambda x: x["elevation_gain"],
        reverse=target_elevation == "max",
    )

    return valid_routes, final_candidates
