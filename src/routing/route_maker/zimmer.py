class Zimmer:

    def __init__(self, graph, config):

        self.graph = graph
        self.config = config

    def validate_step(self, route, node_id):
        """
        Check that node hasn't been visited, or that number of
        overlaps is within parameters
        """
        return

    def generate_steps(self, route):
        """
        Fetch all neighbouring nodes
        Validate each step from current to neighbour
        """
        return

    def generate_potential_routes(self, route):
        """
        Generate list of potential steps
        For each step, new route = route + step
        Build logic for route + step into containers
        """
        return

    def validate_route(self, route):
        """Make sure that a single route is within the parameters set"""

    def generate_incremental_routes(self):
        """
        Generate list of potential routes
        Filter out those which aren't valid
        """
