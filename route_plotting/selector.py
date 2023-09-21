# TODO: Implement class which returns materially different routes from a
#       list of valid routes, use difflib and iterate through the sorted
#       list. Skip any items with a similarity score which is above a defined
#       threshold when compared to any already selected route.
from difflib import SequenceMatcher
from typing import List
from rex_run_planner.containers import Route


class RouteSelector:
    def __init__(
        self, routes: List[Route], num_routes_to_select: int, threshold: float
    ):
        self.routes = routes
        self.n_routes = num_routes_to_select
        self.threshold = threshold

        self.selected_routes: List[Route] = []

    @staticmethod
    def get_similarity(route, selected_route):
        return SequenceMatcher(a=route, b=selected_route).ratio()

    def select_routes(self):
        for route in self.routes:
            is_selectable = True
            route_str = " ".join(str(node) for node in route.route)

            for selected_route in self.selected_routes:
                selected_str = " ".join(
                    str(node) for node in selected_route.route
                )
                route_diff = self.get_similarity(route_str, selected_str)
                if route_diff > self.threshold:
                    is_selectable = False

            if is_selectable:
                if len(self.selected_routes) < self.n_routes:
                    self.selected_routes.append(route)
                if len(self.selected_routes) == self.n_routes:
                    return self.selected_routes

        return self.selected_routes
