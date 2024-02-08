"""Install configuration for the routing package"""

from setuptools import setup

setup(
    name="routing",
    version="0.1.0",
    where="src",
    include="routing",
    setup_requires=[
        "networkx>=3.2.0",
    ],
)
