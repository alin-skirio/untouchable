"""ScrollUp: inside Scrolling, pop the four fingers open and lift them."""


class ScrollUp:
    def __init__(self, width_increase: float = 0.28, min_down: float = 0.035):
        self.width_increase = width_increase
        # Image y grows downward, so a downward flick is a positive travel_y.
        # Anything not clearly down counts as scroll up (including a straight open).
        self.min_down = min_down

    def triggered(self, four_open: bool, width_delta: float, travel_y: float) -> bool:
        return (
            four_open
            and width_delta >= self.width_increase
            and travel_y < self.min_down
        )
