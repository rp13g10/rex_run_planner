import difflib
import random

# from rapidfuzz import fuzz
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


def prune_routes(routes):

    if len(routes) < 6000:
        return routes

    # all_routes = [repr(route["route"]) for route in routes]

    # deduped = process.dedupe(all_routes, threshold=90, scorer=fuzz.ratio)
    # deduped = set(deduped)

    # new_routes = [route for route in routes if repr(route["route"]) in deduped]

    new_routes = []
    for route in tqdm.tqdm(routes, desc="Pruning routes", leave=False):
        is_similar = False
        if not new_routes:
            new_routes.append(route)
            continue
        for new_route in new_routes:
            # similarity = fuzz.ratio(
            #     repr(route["route"]), repr(new_route["route"])
            # )
            match = difflib.SequenceMatcher(route["route"], new_route["route"])
            similarity = match.ratio()
            if 0.8 <= similarity < 1.0:
                is_similar = True
                break
            _ = None
        if not is_similar:
            new_routes.append(route)
        _ = None
    # start_count = len(routes)
    # end_count = len(new_routes)
    # tqdm.tqdm.write(
    #     f"Pruning reduced candidate list from {start_count} to {end_count}"
    # )

    return new_routes


def find_routes(
    lat, lon, max_distance, target_elevation="max", max_candidates=50000
):

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
                    cand_elevation = new_candidate["elevation_gain"]

                    if neighbour == start_pos:
                        if new_candidate["distance"] >= max_distance * 0.95:
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
            candidate_routes = random.sample(candidate_routes, max_candidates)
            # candidate_routes = sorted(
            #     candidate_routes,
            #     key=lambda x: x["elevation_gain"],
            #     reverse=target_elevation == "max",
            # )
            # candidate_routes = candidate_routes[:max_candidates]
            n_candidates = max_candidates
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


valid_routes = find_routes(50.969540, -1.383318, 10.0, max_candidates=100000)

for inx, route in enumerate(valid_routes[:25]):
    fig = plot_route(route)
    dist = route["distance"]
    elev = route["elevation_gain"]
    fig.write_html(f"plots/hilly_{inx}_{dist}_{elev}.html")

for inx, route in enumerate(valid_routes[-25:]):
    fig = plot_route(route)
    dist = route["distance"]
    elev = route["elevation_gain"]
    fig.write_html(f"plots/flat_{inx}_{dist}_{elev}.html")


# docker run -p 6379:6379 -it redis/redis-stack:latest
