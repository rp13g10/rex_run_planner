from dataclasses import dataclass
from typing import List


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
