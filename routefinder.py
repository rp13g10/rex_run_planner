import pickle

from plotly import graph_objects as go

from maputils import (
    fetch_node_coords,
)
from routeutils import find_routes

# TODO: For final output, use difflib to ensure each suggested route is
#       significantly different from the next.

with open("./data/hampshire-latest-compressed.nx", "rb") as fobj:
    osm = pickle.load(fobj)


def plot_route(graph, route):
    lats, lons = [], []
    for node_id in route["route"]:
        lat, lon = fetch_node_coords(graph, node_id)
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


target_distance = 3
target_elevation = "max"
valid_routes, last_candidates = find_routes(
    osm,
    50.969540,
    -1.383318,
    target_distance,
    target_elevation,
    max_candidates=32000,
)

for inx, route in enumerate(last_candidates[:50]):
    fig = plot_route(osm, route)
    dist = route["distance"]
    elev = route["elevation_gain"]
    fig.write_html(f"plots/unfinished/{target_distance}_{inx}_{elev}.html")

if target_elevation == "max":
    for inx, route in enumerate(valid_routes[:50]):
        fig = plot_route(osm, route)
        dist = route["distance"]
        elev = route["elevation_gain"]
        fig.write_html(f"plots/hilly/{target_distance}_{inx}_{elev}.html")

elif target_elevation == "min":
    for inx, route in enumerate(valid_routes[-50:]):
        fig = plot_route(osm, route)
        dist = route["distance"]
        elev = route["elevation_gain"]
        fig.write_html(f"plots/flat/{target_distance}_{inx}_{elev}.html")

with open("data/all_routes.pkl", "wb") as fobj:
    pickle.dump(valid_routes, fobj)

# docker run -p 6379:6379 -it redis/redis-stack:latest
