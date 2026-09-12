"""ScrollDown: inside Scrolling, pop the four fingers open and drop them."""


class ScrollDown:
    def __init__(self, width_increase: float = 0.28, min_down: float = 0.035):
        self.width_increase = width_increase
        self.min_down = min_down

    def triggered(self, four_open: bool, width_delta: float, travel_y: float) -> bool:
        return (
            four_open
            and width_delta >= self.width_increase
            and travel_y >= self.min_down
        )
