"""
Read in a provided OSM data file, convert it to a NetworkX graph which
can be used for path-finding.
"""

import os

curdir = os.path.abspath(os.path.dirname(__file__))

SOURCE_OSM = "data/hampshire-latest.osm"
source_osm = os.path.join(curdir, SOURCE_OSM)

script_loc = os.path.join(curdir, "tools/OsmToRoadGraph/run.py")

os.system(f"python {script_loc} -f {source_osm} -n p --networkx")
