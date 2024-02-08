from dataclasses import dataclass


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
