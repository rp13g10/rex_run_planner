from networkx import Graph
from plotly import graph_objects as go

from rex_run_planner.containers import Route

# TODO: For final output, use difflib to ensure each suggested route is
#       significantly different from the next.


def generate_filename(route: Route) -> str:
    gain = route.elevation_gain
    dist = route.distance
    name = f"gain_{gain:,.2f}_dist_{dist:,.2f}"
    return name


def plot_route(graph: Graph, route: Route) -> go.Figure:
    """For a generated route, generate a Plotly graph which plots it onto
    a mapbox map.

    Args:
        graph (Graph): A graph containing latitude & longitude information for
          every node visited in the provided route
        route (Route): A route generated by a RouteFinder

    Returns:
        go.Figure: A mapbox plot of the provided route
    """
    lats, lons = [], []
    for node_id in route.route:
        lat = graph.nodes[node_id]["lat"]
        lon = graph.nodes[node_id]["lon"]
        lats.append(lat)
        lons.append(lon)

    trace = go.Scattermapbox(mode="lines", lat=lats, lon=lons)

    distance = route.distance
    elevation = route.elevation_gain
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
