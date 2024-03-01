"""This class handles the process of taking steps to increment the current
route."""

from copy import deepcopy
from typing import Iterable, Tuple

from networkx import Graph
from routing.containers.routes import Route, RouteConfig, StepMetrics


class Zimmer:
    """Class which handles stepping from one route to the next. Functions are
    provided which will generate a list of possible nodes to step to, and
    applying those steps to a given route."""

    def __init__(self, graph: Graph, config: RouteConfig):
        """Store down the required information to handle processing of routes

        Args:
            graph (Graph): The network graph representing the geographical
              area in which routes are being generated
            config (RouteConfig): The user-provided configuration for the
              routes to be created
        """

        self.graph = graph
        self.config = config

    def _validate_step(self, route: Route, node_id: int) -> bool:
        """For a given route and potential next step, validate that taking
        this step will not result in the algorithm being unable to return to
        the start point within the configured maximum distance. Steps to
        nodes which have already been visited are also considered invalid,
        except for the last 5% of the route (to increase the odds of being
        able to find a valid circular route).

        Args:
            route (Route): An (incomplete) candidate route
            node_id (int): The ID for the node to be stepped to

        Returns:
            bool: Whether or not the provided node_id would be a valid step
              to take
        """
        # try:
        #     prev_node = route.route[-2]
        # except IndexError:
        #     prev_node = None
        visited = route.visited
        # remaining = self.config.target_distance - route.distance
        # remaining_perc = remaining / (self.config.max_distance)

        if node_id not in visited:
            return True

        first_3_nodes = set(route.route[:3])
        last_3_nodes = set(route.route[-3:])
        if node_id in first_3_nodes and node_id not in last_3_nodes:
            return True
        # elif remaining_perc <= 0.05:
        #     if node_id == prev_node:
        #         return False
        #     return True
        return False

    def generate_possible_steps(self, route: Route) -> Iterable[int]:
        """For a given route, determine which Node IDs are reachable without
        breaching the conditions of the route finding algorithm.

        Args:
            route (Route): An incomplete route

        Returns:
            Iterable[int]: An iterator containing all of the IDs which can
              be stepped to from the current position of the provided route
        """

        cur_node = route.route[-1]
        neighbours = filter(
            lambda node: self._validate_step(route, node),
            self.graph.neighbors(cur_node),
        )

        return neighbours

    def _fetch_step_metrics(self, route: Route, next_node: int) -> StepMetrics:
        """For a candidate route, calculate the change in distance & elevation
        when moving from the end point to the specified neighbour. Record any
        intermediate nodes which are traversed when making this journey.

        Args:
            route (Route): A candidate route
            next_node (int): The ID of a neighbouring node

        Returns:
            StepMetrics: The calculated metrics for this step
        """
        cur_node = route.route[-1]

        step = self.graph[cur_node][next_node]
        distance = step["distance"]
        gain = step["elevation_gain"]
        loss = step["elevation_loss"]
        via = step.get("via", [])

        step_metrics = StepMetrics(
            distance=distance,
            elevation_gain=gain,
            elevation_loss=loss,
            via=via,
        )

        return step_metrics

    def _generate_new_route(self, route: Route, new_id: str) -> Route:
        """Generate a copy of the provided route, giving it a new route ID
        based on the number of candidates & neighbours which have been
        processed so far.

        Args:
            route (Route): A candidate route
            cand_inx (int): The number of candidate routes processed so far
            neigh_inx (int): The number of neighbours processed for the current
              candidate so far

        Returns:
            Route: A copy of the candidate route with an updated route_id
        """
        new_route = deepcopy(route)
        new_route.route_id = new_id

        return new_route

    def _step_to_next_node(
        self, route: Route, next_node: int, step_metrics: StepMetrics
    ) -> Route:
        """For a given route, update its properties to reflect the result of
        taking a step to a neighbouring node

        Args:
            route (Route): A candidate route
            next_node (int): The neighbouring node to step to
            step_metrics (StepMetrics): The impact of making this step

        Returns:
            Route: An updated candidate route, which now ends at 'next_node'
        """

        route.route += step_metrics.via
        route.route.append(next_node)

        route.visited.add(next_node)

        # TODO: Use dunder methods to enable arithmetic with these types
        route.distance += step_metrics.distance
        route.elevation_gain += step_metrics.elevation_gain
        route.elevation_loss += step_metrics.elevation_loss

        return route

    def _validate_route(self, route: Route) -> str:
        """For a newly generated candidate route, validate that it is still
        within the required parameters. If not, then it should be discarded.

        Args:
            route (Route): A candidate route

        Returns:
            str: The status of the route, one of:
              - complete
              - valid
              - invalid
        """
        cur_pos = route.route[-1]

        # Route is too long
        if route.distance >= self.config.max_distance:
            return "invalid"

        # Route cannot be completed without becoming too long
        remaining = self.graph.nodes[cur_pos]["dist_to_start"]
        if (route.distance + remaining) >= self.config.max_distance:
            return "invalid"

        # Route is circular
        start_pos = route.route[0]
        if start_pos == cur_pos:
            # Route is of correct distance
            if (
                (self.config.min_distance)
                <= route.distance
                <= (self.config.max_distance)
            ):
                return "complete"
            else:
                return "invalid"

        return "valid"

    def step_to_next_node(
        self, route: Route, next_node: int, new_id: str
    ) -> Tuple[str, Route]:
        """For a given route and node to step to, perform the step and update
        the route metrics, then validate that the route is still within the
        user-provided parameters.

        Args:
            route (Route): An incomplete route
            next_node (int): The node to be stepped to
            new_id (str): The ID for the new route

        Returns:
            Tuple[str, Route]: The status of the new route (complete, valid or
              invalid), and the new route itself
        """
        # Create a new candidate route
        candidate = self._generate_new_route(route, new_id)

        # Calculate the impact of stepping to the neighbour
        step_metrics = self._fetch_step_metrics(candidate, next_node)

        # Update the new candidate to reflect this step
        candidate = self._step_to_next_node(candidate, next_node, step_metrics)

        candidate_status = self._validate_route(candidate)

        return candidate_status, candidate
