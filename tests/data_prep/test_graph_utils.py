"""
Check that core functionality for graph interactions is working as expected
"""
from unittest.mock import patch
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
    sample_graph.nodes[node_id]["elevation"] = float(node_id)

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
    """Check that coordinate retrieval is functioning properly"""
    # Arrange
    test_node_id = 1
    target = (55.0, 0.0)
    # Act
    result = sample.fetch_node_coords(test_node_id)
    # Assert
    assert result == target


class TestGetElevationCheckpoints:
    """Check that checkpoints are created at the appropriate interval"""

    def test_dist_over_interval(self):
        """
        Check behaviour when A to B is above configured sample distance of 500m
        """
        # Arrange
        test_start_lat = 50.0
        test_start_lon = 0.0
        test_end_lat = 50.01
        test_end_lon = 0.01

        target_lat_checkpoints = [50.0, 50.005, 50.01]
        target_lon_checkpoints = [0.0, 0.005, 0.01]
        target_dist_change = 1.323  # km

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
        """
        Check behaviour when A to B is below configured sample distance of 500m
        """
        # Arrange
        test_start_lat = 50.0
        test_start_lon = 0.0
        test_end_lat = 50.0001
        test_end_lon = 0.0001

        target_lat_checkpoints = [50.0, 50.0001]
        target_lon_checkpoints = [0.0, 0.0001]
        target_dist_change = 0.01323  # km

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
    """Check that elevation is being calculated for every checkpoint and
    summed correctly"""
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


@patch(
    "rex_run_planner.data_prep.graph_utils.GraphUtils._calculate_elevation_change_for_checkpoints"
)
@patch(
    "rex_run_planner.data_prep.graph_utils.GraphUtils._get_elevation_checkpoints"
)
def test_estimate_distance_and_elevation_change(
    mock_get_elevation_checkpoints,
    mock_calculate_elevation_change_for_checkpoints,
):
    """Ensure that data is flowing through the function call in the expected
    manner"""

    # Arrange
    mock_get_elevation_checkpoints.return_value = (
        [0.0, 1.0],
        [0.0, 1.0],
        distance.Distance(kilometers=1.0),
    )
    mock_calculate_elevation_change_for_checkpoints.return_value = (5.0, 10.0)

    test_start_id = 0
    test_end_id = 3

    target_get_elevation_checkpoints_call = (54.0, -1.0, 57.0, 2.0)
    target_calculate_elevation_change_for_checkpoints_call = (
        [0.0, 1.0],
        [0.0, 1.0],
    )

    target_dist_change = 1.0
    target_elevation_gain = 5.0
    target_elevation_loss = 10.0

    # Act
    (
        result_dist_change,
        result_elevation_gain,
        result_elevation_loss,
    ) = sample._estimate_distance_and_elevation_change(
        test_start_id, test_end_id
    )

    # Assert
    mock_get_elevation_checkpoints.assert_called_once_with(
        *target_get_elevation_checkpoints_call
    )
    mock_calculate_elevation_change_for_checkpoints.assert_called_once_with(
        *target_calculate_elevation_change_for_checkpoints_call
    )

    assert target_dist_change == result_dist_change
    assert target_elevation_gain == result_elevation_gain
    assert target_elevation_loss == result_elevation_loss


# sample_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
# sample_graph = graph.Graph(sample_edges)
# coords = {0: (54.0, -1.0), 1: (55.0, 0.0), 2: (56.0, 1.0), 3: (57.0, 2.0)}
# for node_id, (lat, lon) in coords.items():
#     sample_graph.nodes[node_id]["lat"] = lat
#     sample_graph.nodes[node_id]["lon"] = lon

# sample_config = RouteConfig(
#     start_lat=55.0,
#     start_lon=0.0,
#     max_distance=3.0,
#     route_mode="hilly",
#     dist_mode="metric",
#     elevation_interval=500,
# )
# sample = GraphUtils(sample_graph, sample_config)


class TestGetStraightLineDistanceAndElevationChange:
    def test_with_elevation_gain(self):
        """Check that positive elevation changes are handled correctly"""
        # Arrange
        test_start_id = 0
        test_end_id = 3

        # NOTE: Elevation gain set equal to node ID during graph creation
        target_gain = 3.0
        target_loss = 0.0
        target_dist = 384.0

        # Act
        (
            result_dist,
            result_gain,
            result_loss,
        ) = sample._get_straight_line_distance_and_elevation_change(
            test_start_id, test_end_id
        )

        # Assert
        assert target_gain == result_gain
        assert target_loss == result_loss
        assert approx(target_dist, abs=0.1) == result_dist

    def test_with_elevation_loss(self):
        """Check that negative elevation changes are handled correctly"""
        # Arrange
        test_start_id = 3
        test_end_id = 0

        # NOTE: Elevation gain set equal to node ID during graph creation
        target_gain = 0.0
        target_loss = 3.0
        target_dist = 384.0

        # Act
        (
            result_dist,
            result_gain,
            result_loss,
        ) = sample._get_straight_line_distance_and_elevation_change(
            test_start_id, test_end_id
        )

        # Assert
        assert target_gain == result_gain
        assert target_loss == result_loss
        assert approx(target_dist, abs=0.1) == result_dist
