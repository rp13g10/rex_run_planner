from networkx import graph
from geopy import distance
from pytest import approx
from rex_run_planner.containers import RouteConfig
from rex_run_planner.data_prep.graph_utils import GraphUtils

sample_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
sample_graph = graph.Graph(sample_edges)
coords = {0: (54.0, -1.0), 1: (55.0, 0.0), 2: (56.0, 1.0), 3: (57.0, 2.0)}
for node_id, (lat, lon) in coords.items():
    sample_graph.nodes[node_id]["lat"] = lat
    sample_graph.nodes[node_id]["lon"] = lon

sample_config = RouteConfig(
    start_lat=55.0,
    start_lon=0.0,
    max_distance=3.0,
    route_mode="hilly",
    dist_mode="metric",
    elevation_interval=500,
)
sample = GraphUtils(sample_graph, sample_config)


def test_fetch_node_coords():
    # Arrange
    test_node_id = 1
    target = (55.0, 0.0)
    # Act
    result = sample.fetch_node_coords(test_node_id)
    # Assert
    assert result == target


class TestGetElevationCheckpoints:
    def test_dist_over_interval(self):
        # Arrange
        test_start_lat = 50.0
        test_start_lon = 0.0
        test_end_lat = 50.01
        test_end_lon = 0.01

        target_lat_checkpoints = [50.0, 50.005, 50.01]
        target_lon_checkpoints = [0.0, 0.005, 0.01]
        target_dist_change = 1.323

        # Act
        (
            result_lat_checkpoints,
            result_lon_checkpoints,
            result_dist_change,
        ) = sample._get_elevation_checkpoints(
            test_start_lat, test_start_lon, test_end_lat, test_end_lon
        )

        # Assert
        assert approx(result_lat_checkpoints) == target_lat_checkpoints
        assert approx(result_lon_checkpoints) == target_lon_checkpoints
        assert approx(result_dist_change.km, abs=1e-3) == target_dist_change

    def test_dist_below_interval(self):
        # Arrange
        test_start_lat = 50.0
        test_start_lon = 0.0
        test_end_lat = 50.0001
        test_end_lon = 0.0001

        target_lat_checkpoints = [50.0, 50.0001]
        target_lon_checkpoints = [0.0, 0.0001]
        target_dist_change = 0.01323

        # Act
        (
            result_lat_checkpoints,
            result_lon_checkpoints,
            result_dist_change,
        ) = sample._get_elevation_checkpoints(
            test_start_lat, test_start_lon, test_end_lat, test_end_lon
        )

        # Assert
        assert approx(result_lat_checkpoints) == target_lat_checkpoints
        assert approx(result_lon_checkpoints) == target_lon_checkpoints
        assert approx(result_dist_change.km, abs=1e-5) == target_dist_change


def test_calculate_elevation_change_for_checkpoints(monkeypatch):
    # Arrange
    test_lat_checkpoints = [50.0, 49.5, 49.0, 51.0, 52.0, 48.0]
    test_lon_checkpoints = [0.0, -0.5, -1.0, 1.0, 2.0, -2.0]

    monkeypatch.setattr(
        "rex_run_planner.data_prep.graph_utils.get_elevation",
        lambda x, y: x + y,
    )

    target_elevation_gain = 4.0 + 2.0
    target_elevation_loss = 1.0 + 1.0 + 8.0

    # Act
    (
        result_elevation_gain,
        result_elevation_loss,
    ) = sample._calculate_elevation_change_for_checkpoints(
        test_lat_checkpoints, test_lon_checkpoints
    )

    # Assert
    assert result_elevation_gain == target_elevation_gain
    assert result_elevation_loss == target_elevation_loss
