"""Install configuration for the refinement package"""
from setuptools import setup

setup(
    name="rex_run_planner",
    version="0.1.0",
    where="src",
    include="rex_run_planner",
    setup_requires=[
        "relevation",
        "refinement",
        "geopy>=2.4.0",
        "networkx>=3.2.0",
        "plotly>=5.18.0",
        "fuzzywuzzy>=0.18.0",
        "python-levenshtein>=0.23.0"
    ],
)
