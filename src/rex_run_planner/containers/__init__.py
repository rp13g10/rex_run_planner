from typing import List, Set, Optional, Tuple
from dataclasses import dataclass

# TODO: Document the arguments for these classes properly
# TODO: Check whether slots can be used to minimise memory footprint of these
#       classes


@dataclass
class StepMetrics:
    """Container for metrics calculated when stepping from the end of one
    route to a neighbouring node.

    Args:
        distance (float): The distance change
        elevation_gain (float): The elevation increase
        elevation_loss (float): The elevation los
        via (List[int]): The identifiers of any nodes crossed as part of this
          step
    """

    distance: float
    elevation_gain: float
    elevation_loss: float
    via: List[int]


@dataclass
class Route:
    """Container for the information required to represent a single route

    Args:
        route (List[int]): A list of the node IDs which are crossed as part of
          this route, in the order that they are crossed
        visited (Set[int]): A set of all the unique nodes which are visited as
          part of this route
        distance (float): The total distance of the route
        elevation_gain (float): The elevation gain for this route
        elevation_loss (float): The elevation loss for this route
        elevation_gain_potential (float): The elevation gain required in order
          to get back to the route's starting point
        elevation_loss_potential (float): The elevation loss required in order
          to get back to the route's starting point
        ratio (float): The ratio of elevation gained to distance travelled
        terminal_square (Optional[Tuple[int, int]]): The grid square in which
          this route terminates, used only while pruning a list of candidate
          routes"""

    route: List[int]
    visited: Set[int]
    distance: float = 0.0
    elevation_gain: float = 0.0
    elevation_loss: float = 0.0
    elevation_gain_potential: float = 0.0
    elevation_loss_potential: float = 0.0
    ratio: float = 0.0
    route_id: str = "seed"
    terminal_square: Optional[Tuple[int, int]] = None


@dataclass
class RouteConfig:
    """Contains user configuration options for route calculation

    Args:
        start_lat (float): Latitude for the route start point
        start_lon (float): Longitude for the route start point
        max_distance (float): Max distance for the route
        route_mode (str): Set to 'hilly' to generate the hilliest possible
          route, or 'flat' for the flattest possible route
        max_candidates (int): The maximum number of candidate routes which
          should be held in memory. Lower this to increase calculation speed,
          increase it to potentially increase the quality of routes generated.
    """

    start_lat: float
    start_lon: float
    max_distance: float
    route_mode: str
    max_candidates: int = 32000


@dataclass
class BBox:
    """Contains information about the physical boundaries of one or more
    routes

    Args:
        min_lat (float): Minimum latitude
        min_lon (float): Minimum longitude
        max_lat (float): Maximum latitude
        max_lon (float): Maximum longitude"""

    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


@dataclass
class ChainMetrics:
    """Contains information about the distance/elevation change accrued as a
    result of stepping across all edges in a chain of nodes.

    Args:
        start (int): The start node for the provided chain
        end (int): The end node for the provided chain
        gain (float): The elevation gain for the provided chain, in metres
        loss (float): The elevation loss for the provided chain, in metres
        dist (float): The distance travelled over the provided chain, in miles
          or kilometres depending on config.dist_mode
        vias (List[int]): The IDs of any nodes which formed part of the chain,
          but are no longer required as they were of order 2"""

    start: int
    end: int
    gain: float
    loss: float
    dist: float
    vias: List[int]
